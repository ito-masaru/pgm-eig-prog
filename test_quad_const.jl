"""
Intersection of ellipsoids with vanishing constraint

Given m, n, l, sqrtQ[i], c[i], it tries to find x in R^n such that
	* |sqrtQ[i](x-c[i])| <= 1 for all i=1,...,m,
	* l of these inequalities are exact.
"""

using LinearAlgebra
using Random
using Printf
using Plots

include("eig_prog_base.jl")


# Usage: Q, sqrtQ = generate_Q(m,n)
# Q = [Q[1], ..., Q[m]] and Q[i] are n by n
# sqrtQ = [sqrtQ[1], ..., sqrtQ[m]] and Q[i] = sqrtQ[i]^2
function generate_sqrtQ(m,n)
	P = [rand(n,n) for i in 1:m]
	Q = [x*x' for x in P]
	sqrtQ = [sqrt(x) for x in Q]
	return sqrtQ
end

function generate_A_b(J, sqrtQ, c)
	#=
	Returns A and b
	A is an m*(n+1) by n matrix
	b is an element of Jordan algebra
	=#
	
	# c = [c[1], ..., c[m]] c[i] is in R^n
	# A = [sqrtQ[1]; 0; sqrtQ[2]; 0; ..., sqrtQ[m]; 0]
	# b = [-sqrtQ[1]*c[1], 1, ..., -sqrtQ[m]*c[m], 1]
	
	m = length(sqrtQ)
	n = length(c[1])
	A = [sqrtQ[1]; zeros(1,n)]
	b = Float64[]
	append!(b, -sqrtQ[1]*c[1]); append!(b, 1.0)
	for i in 2:m
		A = [A; sqrtQ[i]; zeros(1,n)]
		append!(b, -sqrtQ[i]*c[i]); append!(b, 1.0)
	end
	
	# let b be of Jordan algebraic structure
	b = reshape(J, b)
	
	return A, b
end

"""
Get the function proj_simple_set(y) which returns the projection of y onto range(A)+b
"""
function generate_proj_simple_set(J, A, b)
	# generate basis e_1, ..., e_d in R^(n+1)
	E = nullspace(A') # E[:, i] represents the i-th basis e_i of ker(A')
	d = size(E)[2] # the length of the basis
	e = [reshape(J, E[:, i]) for i in 1:d] # e[i] becomes the basis e_i in the Jordan algebraic structure
	
	function proj_simple_set(y)
		#return y-b-sum( [dot(e[i], y - b) * e[i] for i in 1:d] )
		return y-sum( [dot(e[i], y - b) * e[i] for i in 1:d] )
	end
	return proj_simple_set
end

"""
Get the function proj_eig_set(y) which returns the projection of y (in N1) onto the intersection of N1 and N2 where
	N1 = {v in R^{2m} : v is nonnegative, v is aligned in decreasing order}
	N2 = {v in R^{2m} : last l elements of v are zero}
"""
function generate_proj_eig_set(J, m, l)
	
	function proj_eig_set(x::Array)
		v = Float64[]
	
		for i in 1:(2*m-l)
			push!(v, max(x[i], 0.0))
		end
	
		for i in (2*m-l+1):(2*m)
			push!(v, 0.0)
		end
	
		return v
	end
	
	return proj_eig_set
end


mutable struct EllipTestInstance
	m::Int64
	n::Int64
	sqrtQ::Array
	c::Array
	A::Array
	b::Array
	iter_recorder::BasicPGIterRecorder
	status::Symbol
	x::Array # recovered torajectory
end

function recover_x(y, sqrtQ, c)
	m = length(y)
	x = (1.0/m) * sum([sqrtQ[i] \ y[LORENTZ][i][1:end-1] + c[i] for i in 1:m]) # A \ x computes A^{-1} x
	return x
end

function plot_ellipsoid_2d(sqrtQ, c; label=nothing, plt=nothing)
	"""
	Returns the plot of an ellipsoid |sqrtQ(x-c)| <= 1
	"""
	P = inv(sqrtQ)
	x1(t) = dot(P[1,:], [cos(t), sin(t)]) + c[1]
	x2(t) = dot(P[2,:], [cos(t), sin(t)]) + c[2]
	legend = !isnothing(label)
	if isnothing(plt)
		return plot(x1, x2, 0, 2π, legend=legend, label=label)
	else
		return plot!(plt, x1, x2, 0, 2π, legend=legend, label=label)
	end
end

function plot_ellipsoids_2d(sqrtQ, c)
	"""
	Returns the plot of an ellipsoids |sqrtQ[i](x[i]-c[i])| <= 1 for i=1,2,...
	"""
	plt = plot_ellipsoid_2d(sqrtQ[1], c[1]; label="1")
	for i in 2:length(c)
		plt = plot_ellipsoid_2d(sqrtQ[i], c[i]; label="$i", plt=plt)
	end 
	return plt
end

function plot_result_2d(inst::EllipTestInstance)
	"""
	Return the plot of the result of solve_ellip with n=2.
	"""
	sqrtQ = inst.sqrtQ
	c = inst.c
	
	plt = plot_ellipsoids_2d(sqrtQ, c)
	
	x = inst.x
	x1 = [x[i][1] for i in 1:length(x)]
	x2 = [x[i][2] for i in 1:length(x)]
	plt = plot!(plt, x1, x2, marker=:circle, markersize=2, label="iters $(length(x)-1)")
	return plt
end;

function plot_result_2d(inst_list::Array{EllipTestInstance})
	"""
	Return the plot of the result of solve_ellip with n=2.
	"""
	sqrtQ = inst_list[1].sqrtQ
	c = inst_list[1].c
	
	plt = plot_ellipsoids_2d(sqrtQ, c)
	
	for k in 1:length(inst_list)
		x = inst_list[k].x
		x1 = [x[i][1] for i in 1:length(x)]
		x2 = [x[i][2] for i in 1:length(x)]
		plt = plot!(plt, x1, x2, marker=:circle, markersize=2, label="iters $(length(x)-1)")
	end
	return plt
end;


"""
solve_ellip(l, sqrtQ, c, x_0)

	Given l, sqrtQ, c, x_0, it tries to find a feasible solution to the problem
		* |sqrtQ[i](x-c[i])| <= 1 for all i,
		* l of these inequalities are exact.
	It uses the projected gradient method with the point x_0.
	sqrtQ = [sqrt[1], ..., sqrtQ[m]] and each sqrtQ[i] is a positive definite matrix.
	c = [c[1], ..., c[m]] and each c[i] is of Array{Float64} of the same length as x_0.

Example:
	n = 2; m = 3
	sqrtQ = generate_sqrtQ(m,n)
	c = [zeros(n) for i in 1:m]
	x_0 = zeros(n)
	l = 1
	instance = solve_ellip(l, sqrtQ, c, x_0)
"""
function solve_ellip(l, sqrtQ, c, x_0; report=true, eps=1e-3)
	m = length(sqrtQ)
	n = length(x_0)
	if report; println("m=$m, n=$n, l=$l, eps=$eps"); end
	
	# Define FTvN system J
	J = JAlgebra(); J.Lorentz = (n+1) * ones(Int64, m) # [n+1, n+1, ..., n+1]: m-Lorentz cones
	
	# Define A, b
	A, b = generate_A_b(J, sqrtQ, c)
	
	# Define the projection maps
	proj_simple_set = generate_proj_simple_set(J, A, b)
	proj_eig_set = generate_proj_eig_set(J, m, l)
	
	# Define the initial point
	y_0 = reshape(J, A*x_0) + b
	if report; @printf("x_0=%s,\ny_0=%s\n\n", x_0, y_0[LORENTZ]); end
	
	# Run projected gradient
	if report; println("running projected gradient"); end
	stepsize = 0.99 #
	iter_recorder = BasicPGIterRecorder()
	
	res = solve_feasibility_problem(J, proj_simple_set, proj_eig_set, y_0, stepsize;
		report=report, iter_recorder=iter_recorder, max_time=Inf, max_iter=10000, eps=eps
	)
	
	y_sol = res.solution
	x_sol = recover_x(y_sol, sqrtQ, c)
	
	if report
		@printf("status=%s, ", res.status)
		@printf("iters=%d, ", res.iter_count)
		@printf("err_s=%.3g\n", iter_recorder.err_s[end])
	end
	
	iters_x = [recover_x(y, sqrtQ, c) for y in res.iter_recorder.iters] # recover the iterations
	return EllipTestInstance(m, n, sqrtQ, c, A, b, iter_recorder, res.status, iters_x)
end

function solve_test_instance(l=1; report=true, eps=1e-3)
	"""
	l is the constant for the rank constraint
	"""
	# centers are all the origin
	c = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
	
	# Three ellipsoids
	sqrtQ = [
		[0.698673 0.248792; 0.248792 0.709665],
		[0.362615 0.377577; 0.377577 1.00548],
		[0.275224 0.204515; 0.204515 0.981773]
	]
	
	initial_points = [
		[-3.3, -2.5], [0.2, -2.3], [2.7, -2.3], [3.8, -0.9], [3.8, 1.2],
		[2.0, 1.7], [0.6, 1.7], [-1.0, 2.7], [-2.9, 1.9], [-3.3, 0.3],
		[-0.4, 0.5], [-0.4, 0.2], [-0.3, -0.2], [0.2, -0.3], [0.3, 0.0], [0.3, 0.2], [0.0, 0.0]
	]
	
	results = EllipTestInstance[]
	
	for i in 1:length(initial_points)
		if report; println("\n============ TEST $i ==============\n"); end
		x_0 = initial_points[i]
		res = solve_ellip(l, sqrtQ, c, x_0; report=report, eps=eps)
		push!(results, res);
	end
	
	plt = plot_result_2d(results)
	
	return plt
end

function run_test(; eps=1e-3, save_figure=false)
	plt1 = solve_test_instance(1; eps=eps) # case l=1
	if save_figure; savefig(plt1, "ellip_l1.pdf"); end
	
	plt2 = solve_test_instance(2; eps=eps) # case l=2
	if save_figure; savefig(plt2, "ellip_l2.pdf"); end
	
	plt = plot(plt1, plt2, layout=4)
	display(plt)
end
