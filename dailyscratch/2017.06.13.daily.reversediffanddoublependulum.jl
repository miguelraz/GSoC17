using DifferentialEquations

a = 1
u₀=1/2
f1(t,u) = a*u
tspan = (0.0,1.0)
prob = ODEProblem(f1,u₀,tspan)

sol = solve(prob)

sol = solve(prob,Euler(),dt = 1/2^4)

sol[5]
sol.t[8]

[t+u for (t,u) in tuples(sol)]

[t+2u for (t,u) in zip(sol.t,sol.u)]

function lorenz(t,u,du)
    du[1] = 10.0(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
end

u₀ = [1.0;0.0;0.0]

tspan = (0.0,1.0)

prob = ODEProblem(lorenz,u₀,tspan)

sol = solve(prob)

Pkg.add("Plots")

using ParameterizedFunctions

g = @ode_def LorenzExample begin
    dx = σ*(y-x)
    dy = x*(ρ-z) - y
    dz = x*y - β*z
end σ=>10.0 ρ=>28.0 β=(8/3)

prob = ODEProblem(g,u₀,tspan)
using Plots

sol = solve(prob)

plot(sol)

f(t,u) = 1.01*u
u0=1/2
tspan = (0.0,1.0)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8)
plot(sol,linewidth=5,title="Solution to the linear ODE with a thick line",
     xaxis="Time (t)",yaxis="u(t) (in μm)",label="My Thick Line!") # legend=false

doublependulum = @ode_def doublependulum begin
    Θ1 = l2 * pθ1 - l1 * pΘ2 * cos(Θ1-Θ2)
    Θ2 = -m2 * l2 * pΘ1 * cos(Θ1 - Θ2)
    pΘ1 = - (m1 + m2) * l1 * sin(Θ1) - h1 + h2 * sin(2(Θ1 - Θ2))
    pΘ2 = - m2 * g * l2 * sin(Θ2) + h1 - h2 * sin(2(Θ1 - Θ2))
    h1 = pΘ1 * pΘ2 * sin(Θ1 - Θ2) / (l1 * l2 * [m1 + m2 * square(sin(Θ1))])
    h2 = (m2 * square(l2) * square(pΘ1) + (m1 + m2) * square(l1) * square(pΘ2) - 2*m2*l1*l2*pΘ1*pΘ2*cos(Θ1 - Θ2))/(2*square(l1)*square(l2)*square([m1 + m2 * square(sin(Θ1-Θ2))]))
end l1 => 1.0, l2 => 1.5, m1 => 1, m2 => 3

### ReverseDiffExamples from https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl
# and from here https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/jacobian.jl

Pkg.add("ReverseDiff")
Pkg.add("ForwardDiff")

workspace()
using ReverseDiff: JacobianTape, JacobianConfig, jacobian, jacobian!, compile_jacobian, compile

#########
# setup #
#########

# some objective functions to work with
f(a, b) = (a + b) * (a * b)'
g!(out, a, b) = A_mul_Bc!(out, a + b, a * b)

# pre-record JacobianTapes for `f` and `g` using inputs of shape 10x10 with Float64 elements
const f_tape = JacobianTape(f, (rand(10, 10), rand(10, 10)))
const g_tape = JacobianTape(g!, rand(10, 10), (rand(10, 10), rand(10, 10)))

help(ReverseDiff.TrackedArray)
# compile `f_tape` and `g_tape` into more optimized representations
const compiled_f_tape = compile(f_tape)
const compiled_g_tape = compile(g_tape)

# some inputs and work buffers to play around with
a, b = rand(10, 10), rand(10, 10)
inputs = (a, b)
output = rand(10, 10)
results = (similar(a, 100, 100), similar(b, 100, 100))
fcfg = JacobianConfig(inputs)
gcfg = JacobianConfig(output, inputs)

####################
# taking Jacobians #
####################

# with pre-recorded/compiled tapes (generated in the setup above) #
#-----------------------------------------------------------------#

# these should be the fastest methods, and non-allocating
jacobian!(results, compiled_f_tape, inputs)
jacobian!(results, compiled_g_tape, inputs)

# these should be the second fastest methods, and also non-allocating
jacobian!(results, f_tape, inputs)
jacobian!(results, g_tape, inputs)

# with a pre-allocated JacobianConfig #
#-------------------------------------#
# this is more flexible than a pre-recorded tape, but can be wasteful since the tape
# will be re-recorded for every call.

jacobian!(results, f, inputs, fcfg)

jacobian(f, inputs, fcfg)

jacobian!(results, g!, output, inputs, gcfg)

jacobian(g!, output, inputs, gcfg)

# without a pre-allocated JacobianConfig #
#----------------------------------------#
# convenient, but pretty wasteful since it has to allocate the JacobianConfig itself

jacobian!(results, f, inputs)

jacobian(f, inputs)

jacobian!(results, g!, output, inputs)

jacobian(g!, output, inputs)

workspace()

using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile

#########
# setup #
#########

# some objective function to work with
f(a, b) = sum(a' * b + a * b')

# pre-record a GradientTape for `f` using inputs of shape 100x100 with Float64 elements
const f_tape = GradientTape(f, (rand(100, 100), rand(100, 100)))

# compile `f_tape` into a more optimized representation
const compiled_f_tape = compile(f_tape)

# some inputs and work buffers to play around with
a, b = rand(100, 100), rand(100, 100)
inputs = (a, b)
results = (similar(a), similar(b))
all_results = map(DiffBase.GradientResult, results)
cfg = GradientConfig(inputs)

####################
# taking gradients #
####################

# with pre-recorded/compiled tapes (generated in the setup above) #
#-----------------------------------------------------------------#

# this should be the fastest method, and non-allocating
gradient!(results, compiled_f_tape, inputs)

# the same as the above, but in addition to calculating the gradients, the value `f(a, b)`
# is loaded into the the provided `DiffResult` instances (see DiffBase.jl documentation).
gradient!(all_results, compiled_f_tape, inputs)

# this should be the second fastest method, and also non-allocating
gradient!(results, f_tape, inputs)

# you can also make your own function if you want to abstract away the tape
∇f!(results, inputs) = gradient!(results, compiled_f_tape, inputs)

# with a pre-allocated GradientConfig #
#-------------------------------------#
# these methods are more flexible than a pre-recorded tape, but can be
# wasteful since the tape will be re-recorded for every call.

gradient!(results, f, inputs, cfg)

gradient(f, inputs, cfg)

# without a pre-allocated GradientConfig #
#----------------------------------------#
# convenient, but pretty wasteful since it has to allocate the GradientConfig itself

gradient!(results, f, inputs)

gradient(f, inputs)

@show gradient(f,inputs)
