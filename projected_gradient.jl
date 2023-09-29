"""
projected_gradient.jl
projected gradient method for feasibility problems
"""

abstract type AbstractPGIterRecorder end # objects for which `record_iter!` and `report_iter` are defined

"""BasicPGIterRecorder"""
abstract type AbstractBasicPGIterRecorder <: AbstractPGIterRecorder end

mutable struct BasicPGIterRecorder <: AbstractBasicPGIterRecorder
	iters::Array # list of x_k for k >= 0
	cputime_us::Array{UInt64} # list of CPU time in microseconds
	err_s::Array # list of distance from x_k to simple set
	max_iter_storage::Union{Int64,Nothing} # the maximal length of iters; setting `nothing` will store all of iterates
	err_opt::Array # list of |x_k - opt_sol| (available if opt_sol is known)
	opt_sol::Union{Array,Nothing} # optimal solution (if known)
end


function BasicPGIterRecorder(; max_iter_storage=nothing, opt_sol=nothing)
	r = BasicPGIterRecorder([], UInt64[], Float64[], max_iter_storage, Float64[], opt_sol)
	return r
end

function record_iter!(r::BasicPGIterRecorder, k::Int64, x::Array, x_s::Array, y::Union{Array,Nothing}, x_prev::Union{Array,Nothing})
	push!(r.iters, x)
	push!(r.cputime_us, CPUTime.CPUtime_us())
	push!(r.err_s, norm(x-x_s))
	
	if !isnothing(r.opt_sol)
		push!(r.err_opt, norm(x-r.opt_sol))
	end
	
	if !isnothing(r.max_iter_storage) && length(r.iters) > r.max_iter_storage
		deleteat!(r.iters, 1) # delete the first (oldest) element
	end
end

function report_iter(r::BasicPGIterRecorder)
	iter_count = length(r.err_s)-1
	err_s = r.err_s[end]
	
	if length(r.iters) == 1
		cputime_so_far = 0.0
		cputime_delta = NaN
		delta = NaN
	else
		cputime_so_far = (r.cputime_us[end] - r.cputime_us[1])/10^6
		cputime_delta = (r.cputime_us[end] - r.cputime_us[end-1])/10^6
		delta = isnothing(r.iters[end-1]) ? NaN : norm(r.iters[end] - r.iters[end-1])
	end

	@printf("iter=%3d, time=%.2gs(+%.2f), err_s=%.3g, delta=%.3g",
		iter_count, cputime_so_far, cputime_delta, err_s, delta
	)
	if !isnothing(r.opt_sol); @printf(", erropt=%.3g"); end
	@printf("\n")
end

"""ProjGradResult"""
struct ProjGradResult
	status::Symbol					# either :finished, :epsdiff, :max_iter, :max_time
	iter_count::Int64				# >= 0
	cputime::Float64				# running time in second
	solution::Union{Array,Nothing}	# the final iterate x_k
	iter_recorder::AbstractPGIterRecorder
end

"""
	Apply projected gradient method to the feasibility problem
		x in S1 and lambda(x) in S2
	where
		S1 is a simple set,
		lambda(x) is the eigenvalue map
		S2 is **contained** in the range of lambda (namely, lambda(lambda^{-1}(S2)) = S2 holds).
	
"""
function solve_feasibility_problem(
	FTvN_system::Union{JAlgebra,Array{JAlgebra}},	# Jordan algebra or an array of Jordan algebra
	proj_simple_set::Function,		# x -> projection of x (in FTvN system) onto the simple set S1
	proj_eig_set::Function,			# v -> projection of v (in the range of lambda) onto the S2
	x_0::Array,						# the initial point
	stepsize::Float64				# in (0,1]
	;
	report::Bool=false,
	max_time::Union{Float64, Nothing}=nothing, # max time in second
	max_iter::Union{Int64, Nothing}=nothing, # max iteration
	delta_eps=1e-16,	# terminate if |x_{k-1}-x_k| <= delta_eps
	eps=1e-4,			# terminateif |x_k - proj_S1(x_k)| <= eps (when `stop_criterion` is not specified)
	iter_recorder=nothing,
	stop_criterion=nothing
	)
	
	# initialization
	if isnothing(stop_criterion)
		default_stop_criterion(x, x_s, y, x_prev, stepsize, iter_recorder) = !isnothing(x_prev) && (norm(x-x_s) <= eps)
		stop_criterion = default_stop_criterion
	end
	
	status = :running
	x_prev = nothing	# the previous iterate x_{k-1}
	x = x_0				# the current iterate x_k
	x_s = nothing		# the projection of x_k onto the simple set
	y = nothing			# y_{k-1} = convex comination of x_{k-1} and its projection onto the simple set
	delta = nothing		# the distance between x_k and x_{k-1}
	k = 0 				# iteration counter
	start_cputime = CPUTime.CPUtime_us()
	
	while true
		x_s = proj_simple_set(x)
		
		# record/report the iteration
		if !isnothing(iter_recorder)
			record_iter!(iter_recorder, k, x, x_s, y, x_prev)
		end
		if report; report_iter(iter_recorder); end
		
		# check stop criteria
		cputime_sec = (CPUTime.CPUtime_us()-start_cputime)*1e-6
		if !isnothing(x_prev) && norm(x-x_prev) < delta_eps
			status = :epsdiff
			break
		elseif stop_criterion(x, x_s, y, x_prev, stepsize, iter_recorder)
			status = :finished
			break
		elseif !isnothing(max_iter) && k >= max_iter
			status = :max_iter
			break
		elseif !isnothing(max_time) && cputime_sec > max_time
			status = :max_time
			break
		end
		
		# update iteration
		x_prev = x
		y = (1-stepsize) * x + stepsize * x_s
		x = proj_spec(FTvN_system, proj_eig_set, y)
		k += 1
	end
	
	cputime_sec = (CPUTime.CPUtime_us()-start_cputime)*1e-6
	return ProjGradResult(status, k, cputime_sec, x, iter_recorder)
end

