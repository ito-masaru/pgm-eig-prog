# Projected gradient method for eigenvalue programming

This is an implementation of numerical experiments in the paper ``[Eigenvalue programming beyond matrices](https://optimization-online.org/?p=24315)'' by Masaru Ito and Bruno Lourenço.

All experiments are implemented in Julia 1.6.7 and can be examined as follows.

1. Run Julia by `julia --project=/full_path/to/pgm-eig-prog` where `pgm-eig-prog` is the directory containing the source files.
2. (Optinal) One can resolve the dependencies by switching to the Pkg mode (type `]`) and executing the command `instantiate`.
3. The numerical experiment in Section 5.1 can be examined as follows.
	
	```
	include("/full_path/to/pgm-eig-prog/test_inv_eigval.jl")
	run_test()
	```

	`run_test()` has optional arguments that can be seen in the file `test_inv_eigval.jl`. By default 
it will output the results of each test.

	
4. The numerical experiment in Section 5.2 can be examined as follows.
	
	```
	include("/full_path/to/pgm-eig-prog/test_quad_const.jl")
	run_test()
	```
