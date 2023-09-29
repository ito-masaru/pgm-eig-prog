"""

algebra.jl

Basic functions for Jordan algebra and FTvN system

Example of defining Jordan algebra:
	J = JAlgebra(10,[3,4],[5,5])

	J = JAlgebra()
	J.Semidefinite=[3,4]

Elements in Jordan algebra is expressed as a nested array
Example:
	x = zero(J) genrates the zero element in the Jordan algebra J
		x[FREE] is a vector for free part
		x[LORENTZ][i] is a vector of size J.Lorentz[i]
		x[SEMIDEFINITE][i] is a symmetric matrix of size J.Semidefinite[i]

Elements in FTvN system is expressed as either a Jordan algebra or an array of Jordan algebras
Example:
	J1 = JAlgebra(); J1.Lorentz = [3,3];
	J2 = JAlgebra(); J2.Lorentz = [4,4];
	FTvN_system = [J1,J2];
	x = zero(FTvN_system); # x[1] and x[2] are zero elements of J1 and J2, respectively.
"""
mutable struct JAlgebra
	Free::Int64
	Lorentz::Array{Int64}
	Semidefinite::Array{Int64}
end

JAlgebra() = JAlgebra(0, Int64[], Int64[])
JAlgebra_SOC(d::Int64) = JAlgebra(0, [d], Int64[])
JAlgebra_Sym(d::Int64) = JAlgebra(0, Int64[], [d])


"""
Index of category in the Jordal algebra
"""
const FREE = 1
const LORENTZ = 2
const SEMIDEFINITE = 3


function has_only_Free(J::JAlgebra)
	return J.Free != 0 && length(J.Lorentz) == 0 && length(J.Semidefinite) == 0
end

function has_only_Lorentz(J::JAlgebra)
	return J.Free == 0 && length(J.Lorentz) > 0 && length(J.Semidefinite) == 0
end

function has_only_Semidefinite(J::JAlgebra)
	return J.Free == 0 && length(J.Lorentz) == 0 && length(J.Semidefinite) > 0
end

"""returns the rank of the Jordan algebra"""
function rank(J::JAlgebra)
	return 2*length(J.Lorentz) + sum(J.Semidefinite) # also works for empty cases: length(Int64[])==sum(Int64[])==0
end

"""returns the length of vectrized elements of the Jordan alagebra `J`"""
function veclen(J::JAlgebra)
	return J.Free + sum(J.Lorentz) + sum([d*d for d in J.Semidefinite])
end

"""returns the length of vectrized elements of the FTvN system `F`"""
function veclen(F::Array{JAlgebra})
	return sum([veclen(J) for J in F])
end

"""returns the dimension of the Jordan algebra `J`"""
function dim(J::JAlgebra)
	return Int(J.Free + sum(J.Lorentz) + sum([d*(d+1)/2 for d in J.Semidefinite]))
end

"""returns the dimension of the FTvN system `F`"""
function dim(F::Array{JAlgebra})
	return sum([dim(J) for J in F])
end

"""returns the zero element in the Jordal algebra `J` in the vectorized form"""
function zero_vectorized(J::JAlgebra)
	return zeros(veclen(J))
end

"""returns the zero element in the FTvN system `F` in the vectorized form"""
function zero_vectorized(F::Array{JAlgebra})
	return zeros(veclen(F))
end

"""returns the zero element in the Jordan algebra `J` in the structured form"""
function zero(J::JAlgebra)
	return reshape(J, zeros(veclen(J)))
end

function zero(FTvN_system::Array{JAlgebra})
	return [zero(J[i]) for i in 1:length(FTvN_system)]
end

"""returns a random element of the Jordan algebra `J`"""
function randJ(J::JAlgebra)
	x = zero(J)
	if J.Free != 0
		x[FREE] = rand(Float64, J.Free)
	end

	if J.Lorentz != Int64[]
		for j in 1:length(J.Lorentz)
			x[LORENTZ][j] = rand(Float64, J.Lorentz[j])
		end
	end

	if J.Semidefinite != Int64[]
		for j in 1:length(J.Semidefinite)
			d = J.Semidefinite[j]
			y = rand(Float64, (d,d))
			x[SEMIDEFINITE][j] = (y+y')/2
		end
	end
	
	return x
end

"""
returns a random element of the FTvN system F
"""
function rand_FTvN(F::Array{JAlgebra})
	return [randJ(F[i]) for i in 1:length(F)]
end

function rand_FTvN(F::JAlgebra)
	return randJ(F)
end

#-----------------
# functions for structure conversion
#-----------------


"""
vectorize(J,x): convert the element `x` of the Jordan algebra `J` from the structured form into the vectorized form
"""
function vectorize(J::JAlgebra, x::AbstractArray)
	v = Float64[];

	if J.Free != 0
		append!(v, x[FREE])
	end

	for i in 1:length(J.Lorentz)
		append!(v, x[LORENTZ][i])
	end

	for i in 1:length(J.Semidefinite)
		append!(v, vec(x[SEMIDEFINITE][i]))
	end

	return v
end

"""
vectorize(F,x): convert the element `x` of the FTvN system `F` from the structured form into the vectorized form
"""
function vectorize(F::Array{JAlgebra}, x::AbstractArray)
	v = Float64[]
	for i=1:length(F)
		append!(v, vectorize(F[i], x[i]))
	end
	return v
end


"""
reshape(J,v): convert the element `x` of the Jordan algebra `J` from the vector form into the structured form
"""
function reshape(J::JAlgebra, v::Array{Float64})
	x = []::Array{Any}
	i = 1 # the current index
	if J.Free == 0
		append!(x, 0.0)
	else
		push!(x, v[i:i+J.Free-1])
		i += J.Free
	end

	if J.Lorentz == Int64[]
		push!(x, 0.0)
	else
		block = []
		for j in 1:length(J.Lorentz)
			push!(block, v[i:i+J.Lorentz[j]-1])
			i += J.Lorentz[j]
		end
		push!(x, block)
	end

	if J.Semidefinite == Int64[]
		push!(x, 0.0)
	else
		block = []
		for j in 1:length(J.Semidefinite)
			d = J.Semidefinite[j]
			push!(block, reshape(v[i:i+d*d-1], (d,d)))
			i += d*d
		end
		push!(x, block)
	end

	return x
end

"""
reshape(F,v): convert the element `v` of the FTvN system `F` from the vector form into the structured form
"""
function reshape(F::Array{JAlgebra}, v::Array{Float64})
	x = []::Array{Any}
	i = 1 # current index
	for k=1:length(F)
		J = F[k]
		d = veclen(J)
		push!(x, reshape(J, v[i:i+d-1]))
		i += d
	end
	return x
end

#-----------------
# functions for the spectral decomposition
#-----------------


"""
eigL:
spectral decomposition associated with Lorentz cone
"""

function eigL(v::Array, t::Float64)
	"""Returns the spectral decomposition of (v,t)"""
	norm_v = norm(v)
	eps = 1.0e-8
	if norm_v < eps
		w = zeros(length(v)); w[1] = 1;
		v_max = sqrt(2)/2 * vcat(w, 1)
		v_min = sqrt(2)/2 * vcat(-w, 1)
	else
		v_max = sqrt(2)/2 * vcat(v/norm_v, 1)
		v_min = sqrt(2)/2 * vcat(-v/norm_v, 1)
	end
	lambda_max = sqrt(2)/2 * (t+norm_v)
	lambda_min = sqrt(2)/2 * (t-norm_v)
	return ((lambda_max, lambda_min), (v_max, v_min))
end

function eigL(x::Array)
	return eigL(x[1:end-1], x[end])
end


"""
eigJ(J,x)
	* Returns the spectral decomposition of the element `x` of the Jordan algebra `J` in the simple form.
	* Returns `(eigvals, idempotents)`
	* `eigvals` is an array of eigenvalues of `x` in the decending order.
	* `idempotents` is the list of pairs `(idempotent, index_in_JAlgebra)`.
	* `index_in_JAlgebra` is of the form `(Category, j)` where
		- `Category` is equal to either `LORENTZ` or `SEMIDEFINITE`.
		- `j` is the index in the category. `j==0` means that this category is single.
"""
function eigJ(J::JAlgebra, x::Array)
	eigen_values = Float64[]
	idempotents = [] # the list of pairs (idempotent, index_in_JAlgebra)

	# decompose Lorentz part
	for i in 1:length(J.Lorentz)
		y = x[LORENTZ][i]
		evals, evecs = eigL(y)
		append!(eigen_values, evals)
		for j in 1:length(evecs)
			append!(idempotents, [(evecs[j], (LORENTZ,i))])
		end
	end

	# decompose Semidefinite part
	for i in 1:length(J.Semidefinite)
		y = x[SEMIDEFINITE][i]
		vals, vecs = eigen((y+y')/2, sortby = x -> -x) # descending order
		append!(eigen_values, vals)
		for j in 1:length(vals)
			idemp = vecs[:,j] * vecs[:,j]'
			append!(idempotents, [(idemp, (SEMIDEFINITE,i))])
		end
	end
	
	if length(J.Lorentz) + length(J.Semidefinite) > 1
		perm = sortperm(eigen_values, rev=true)
		return eigen_values[perm], idempotents[perm]
	else
		return eigen_values, idempotents
	end
end

"""
pullbackJ(J, eigvals, idempotents)
	* reconstruct the element of the Jordan algebra `J` (in the simple form)
	  from the spectral decomposition specified by `(eigvals, idempotents)`.
	* `(eigvals, idempotents)` is of the same format as the return value of `eigJ`.
"""
function pullbackJ(J::JAlgebra, eigvals::Array, idempotents::Array)
	x = zero(J)

	for i in 1:length(eigvals)
		idemp, (cate, j) = idempotents[i]
		x[cate][j] += eigvals[i] * idemp
	end

	return x
end


"""
eigF(F, x)
	* compute the eigen-decomposition of x
	* returns lambda_x, idemp_x
	* lambda_x[i] is the eigenvalues of x[i]
	* idemp_x can be used to call pullbackF
"""
function eigF(F::Array{JAlgebra}, x::Array)
	lambda_x = []
	idemp_x = []
	
	for i in 1:length(F) # length of the direct product in the FTvN system
		lambda_x_i, idemp_x_i = eigJ(F[i], x[i])
		push!(lambda_x, lambda_x_i)
		push!(idemp_x, idemp_x_i)
	end
	
	return lambda_x, idemp_x
end

function eigF(F::JAlgebra, x::Array)
	return eigJ(F, x)
end

"""
pullbackF(F, lambda_x, idemp_x): reconstruct x so that eigF(F, x) returns (lambda_x, idemp_x)
"""
function pullbackF(F::Array{JAlgebra}, lambda_x::Array, idemp_x::Array)
	recover_x = []
	for i in 1:length(F)
		recover_x_i = pullbackJ(F[i], lambda_x[i], idemp_x[i])
		push!(recover_x, recover_x_i)
	end
	return recover_x
end

function pullbackF(J::JAlgebra, eigvals::Array, idempotents::Array)
	return pullbackJ(J, eigvals, idempotents)
end

# projection of a point onto the spectral set
function proj_spec(J::JAlgebra, proj_eig_set::Function, x::Array)
	lambda_x, idemp_x = eigJ(J, x)	
	proj_lambda_x = proj_eig_set(lambda_x)
	proj_x = pullbackJ(J, proj_lambda_x, idemp_x)
	proj_x[FREE] = x[FREE] # restore the free part	
	return proj_x
end

# projection of a point onto the spectral set
function proj_spec(FTvN_system::Array{JAlgebra}, proj_eig_set::Function, x::Array)
	lambda_x, idemp_x = eigF(FTvN_system, x)
	proj_lambda_x = proj_eig_set(lambda_x)
	return pullbackF(FTvN_system, proj_lambda_x, idemp_x)
end
