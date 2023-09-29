"""
Numerical experiment for inverse eigenvalue problem:
	Find x in E such that x in L and lambda(x) = true_lambda
where
	E: Jordan Algebra associated with Lorentz(n+1)^m * Semidefinite(n)
	L = A_0 + range(A) is a an affine space in E (randomly generated); A: R^d -> E is a linear map
"""

using LinearAlgebra
using Random
using Printf
using JLD
using Statistics
using Plots

include("eig_prog_base.jl")

function generate_random_problem(m,n,d; eig_type=:blockwise)
	"""
	m: number of Lorentz cone
	n: variable dimension will be (n+1)*m + n*(n+1)/2
	d: dimension of affine space L
	eig_type: :blockwise or :ordered
	"""
	
	# generate FTvN system L^{n+1} * ... * L^{n+1} (m times) * S^n
	if eig_type == :blockwise
		FTvN_system = [JAlgebra_SOC(n+1) for i=1:m]
		push!(FTvN_system, JAlgebra_Sym(n))
	elseif eig_type == :ordered
		F = JAlgebra(); # empty Jordan algebra
		F.Lorentz = (n+1) * ones(Int64, m); # [n+1, n+1, ..., n+1] of length m
		F.Semidefinite = [n]; # S^n
	end
	
	# generate a linear map A: R^d -> FTvN_system
	# A = [A_1,...A_d] where A_i is a random element in the FTvN system
	A = [rand_FTvN(FTvN_system) for i=1:d]
	
	# generate true_lambda
	u = rand(d); # random element of R^d
	Au = sum([u[i]*A[i] for i in 1:length(A)]); # a random image A(u)=A_1 u_1 + ... + A_d u_d
	A_0 = rand_FTvN(FTvN_system)
	x_opt = A_0 + Au
	true_lambda, idemp = eigF(FTvN_system, x_opt)
	
	return (FTvN_system, A, A_0, true_lambda, x_opt);
end


"""
projection onto the affine space A_0 + range(A)
	A = [A_1,...,A_d]
	A_1,...,A_d and A_0 are elements of the FTvN system F
"""
function generate_proj_simple_set(F, A, A_0)
	# vectorize the linear map A
	d = length(A)
	vec_A = Float64[]
	for i in 1:d
		append!(vec_A, vectorize(F, A[i]))
	end
	ncol = veclen(F) # the number of columns of adj(A)
	mat_A = reshape(vec_A, ncol, d) # d by ncol matrix expression of A
	vec_A_0 = vectorize(F, A_0)
	
	# construct the matrix form of the adjoint map of A
	mat_A_adj = Matrix{Float64}(transpose(mat_A)) # matrix expression of the adjoint of A
	E_vec = nullspace(mat_A_adj); # E[:, i] represents the i-th basis e_i for ker(A^*) = orthcomp(range(A))
	r = size(E_vec)[2] # the length of the basis
	E = [reshape(F, E_vec[:, i]) for i in 1:r] # E[i] becomes the basis e_i in the FTvN system
	
	function proj_simple_set(x)
		g = sum( [dot(E[i], x - A_0) * E[i] for i in 1:r] )
		return x - g
	end
	
	function recover_c(x)
		proj_x = vectorize(F, proj_simple_set(x))
		return mat_A \ (proj_x - vec_A_0) # solution for A_0 + c[1]A[1] + ... + c[d]A[d] = proj_x
	end
	
	return proj_simple_set, recover_c
end

# instance structure of inverse eigenvalue problem
mutable struct InvEigInstance
	m::Int64 # number of Jordan algebra associated to Lorentz cone
	n::Int64 # n+1 is the dimension of each Lorentz cone
	d::Int64 # the dimension of input variable in the inverse eigenvalue problem
	seed_value::Int64 # to identify random instance
	FTvN_system::Union{JAlgebra,Array{JAlgebra}}
	A::Array
	A_0::Array
	true_lambda::Array
	x_opt::Array
	x_0::Array
	iter_recorder::BasicPGIterRecorder # detail of projected gradient
	status::Symbol # returned by projected gradient
	iter_count::Int64 # the number of iterations of the projected gradient
	cputime::Float64  # the running time of projected gradient
	retry::Int64 # the number restart of the projected gradient
	rel_dist::Float64 # final relative distance from the initial point to the optimal solution
end


function test_inv_eig(m,n,d; stepsize=0.99, eps=1e-3, eig_type=:blockwise, report=true, report_alg=false, max_retry=20, max_iter=10000, seed_value=nothing)
	if !isnothing(seed_value)
		Random.seed!(seed_value);
	end
	FTvN_system, A, A_0, true_lambda, x_opt = generate_random_problem(m, n, d; eig_type=eig_type)
	proj_simple_set, recover_c = generate_proj_simple_set(FTvN_system, A, A_0)
	proj_eig_set(lambda::Array) = true_lambda # projection onto the singleton {true_lambda}
	
	retry = 0 # run proj grad at most max_retry times
	unit_elem = rand_FTvN(FTvN_system);
	unit_elem = unit_elem / norm(unit_elem);
	rel_dist_opt = 100; # relative distance from x_0 to x_opt
	
	x_0 = x_opt + unit_elem*rel_dist_opt*norm(x_opt);
	res = solve_feasibility_problem(FTvN_system, proj_simple_set, proj_eig_set, x_0, stepsize;
		max_iter=max_iter, eps=eps, iter_recorder=BasicPGIterRecorder(opt_sol=x_opt) #,report=true
	)
	
	while res.status != :finished && retry < max_retry
		# retry
		rel_dist_opt /= 2; # make initial point closer to x_opt
		x_0 = x_opt + unit_elem*rel_dist_opt*norm(x_opt);
		res = solve_feasibility_problem(FTvN_system, proj_simple_set, proj_eig_set, x_0, stepsize;
			report=report_alg, max_iter=max_iter, eps=eps, iter_recorder=BasicPGIterRecorder(opt_sol=x_opt)
		)
		retry += 1;
	end
	
	inst =	InvEigInstance(
				m, n, d, seed_value, FTvN_system, A, A_0, true_lambda,
				x_opt, x_0, res.iter_recorder, res.status, res.iter_count, res.cputime, retry, rel_dist_opt
			)
	
	return inst
end

function get_file_name(m,n,d,eig_type,stepsize,eps,max_iter)
	return @sprintf("inveig_m=%d_n=%d_d=%d_s=%.3f_e=%g_eig=%s_maxi=%d", m,n,d,stepsize,eps,eig_type,max_iter)
end

function load_result(m::Int64, n::Int64, d::Int64; save_dir="data", eig_type=:blockwise, stepsize=0.99, eps=1e-3, max_iter=10000)
	save_path = joinpath(save_dir, get_file_name(m,n,d,eig_type,stepsize,eps,max_iter)*".jld")
	return load(save_path, "results")
end

function show_test_results(m::Int64, n::Int64, d::Int64;
	results=nothing, save_dir="data", eig_type=:blockwise, stepsize=0.99, eps=1e-3, max_iter=10000)
	
	summary_str(vals) = @sprintf("ave=%.3g, max=%d, min=%d, std=%.3g", mean(vals), maximum(vals), minimum(vals), Statistics.std(vals))
	
	if isnothing(results)
		results = load_result(m,n,d; save_dir=save_dir, eig_type=eig_type, stepsize=stepsize, eps=eps, max_iter=max_iter)
	end
	
	retry_hist = [inst.retry for inst in results]
	iter_hist = [inst.iter_count for inst in results]
	
	print("Finished: m=$m, n=$n, d=$d, eig_type=$eig_type, stepsize=$stepsize, eps=$eps, max_iter=$max_iter, ")
	@printf("iters=(%s), restarts=(%s)\n", summary_str(iter_hist), summary_str(retry_hist))
end

function run_test(;
		stepsize		= 0.99,
		eps				= 1e-3,
		max_iter		= 10000,
		skip_finished	= false, # skip test if .jld data file was found
		report			= true,  # print the report of each test results
		report_alg		= false, # print the report of iterations of projected gradient
		save_result		= false, # save test results in current_path/save_dir/***.jld
		save_dir		= "data",
		num_test_inst	= 10,
		eig_types		= [:blockwise, :ordered], # list of types of FTvN systems to be tested
		params			= [(m,n,d_ratio) for
								d_ratio in [0.2*i for i=1:4],
								n in [10],
								m in [0,1,5]
						]
	)
	
	for eig_type in eig_types
		for (m, n, d_ratio) in params
			dimension = m*(n+1) + n*(n+1)/2
			d = Int(floor(dimension * d_ratio))
		
			save_path = joinpath(save_dir, get_file_name(m,n,d,eig_type,stepsize,eps,max_iter)*".jld")
			if skip_finished && isfile(save_path)
				results = load_result(m,n,d; eig_type=eig_type, stepsize=stepsize, eps=eps, max_iter=max_iter)
				show_test_results(m,n,d; results=results, stepsize=stepsize, eps=eps, max_iter=max_iter)
				continue
			end
		
			if report
				println("\n=== TEST m=$m, n=$n, d=$d, stepsize=$stepsize, eps=$eps, eig_type=$eig_type ===")
			end
			results = []
		
			for i = 1:num_test_inst
				seed_value = i + m + n
				inst = test_inv_eig(m,n,d; stepsize=stepsize, eps=eps, report=report, max_iter=max_iter, seed_value=seed_value)
				push!(results, inst);
				if report
					rec = inst.iter_recorder
					@printf("[instance %d] ", i)
					@printf("status=%s, ", inst.status)
					@printf("iters=%d, ", inst.iter_count)
					@printf("err_s=%.3g, ", rec.err_s[end])
					@printf("err_opt=%.3g, ", rec.err_opt[end])
					@printf("cputime=%.3g, ", inst.cputime)
					@printf("rel_dist=%.3g, ", inst.rel_dist)
					@printf("retry=%d", inst.retry)
		
					@printf("\n");
				end
			end
		
			if save_result
				save(save_path, "results", results)
			end
		
			show_test_results(m,n,d; results=results, stepsize=stepsize, eps=eps, max_iter=max_iter)
		end
	end
end