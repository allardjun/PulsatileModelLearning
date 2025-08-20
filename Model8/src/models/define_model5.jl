
struct Model5 <: AbstractClassicalModel
    p_classical_derepresented_ig::ComponentArray{Float64}
    p_derepresented_lowerbounds::ComponentArray{Float64}
    p_derepresented_upperbounds::ComponentArray{Float64}
    u0::Vector{Float64}
    params_repr_ig::ComponentArray{Float64}
    params_derepresented_ig::ComponentArray{Float64}
end

function make_Model5()
    p_classical_derepresented_ig = ComponentArray{Float64}(
        tau_a=5.0f0, 
        tau_m=50.0f0, 
        tau_c=100.0f0, 
    )

    p_derepresented_lowerbounds = ComponentArray{Float64}(
        tau_a=1.0f0,
        tau_m=1.0f0,
        tau_c=100.0f0 * 1e-2,
    )

    p_derepresented_upperbounds = ComponentArray{Float64}(
        tau_a=5.0f0 * 1e+1,
        tau_m=50.0f0 * 1e+3,
        tau_c=100.0f0 * 1e+3,
    )

    # initial condition
    u0 = [0.0, 0.0, 0.0, 0.0]

    params_repr_ig = ComponentArray( p_classical=represent_on_type(p_classical_derepresented_ig, Model5))

    params_derepresented_ig = ComponentArray( p_classical=p_classical_derepresented_ig)

    return Model5(
        p_classical_derepresented_ig,
        p_derepresented_lowerbounds,
        p_derepresented_upperbounds,
        u0,
        params_repr_ig,
        params_derepresented_ig,
    )
end

# returns rhs, suitable for inclusion in ODEProblem
function make_rhs(i_func, model::Model5)

    # linear architecture with production-promotion

    function rhs(du, u, p_all_derepresented, t)
        a, m, w, c = u
        ta, tm, tc = p_all_derepresented.p_classical

        du[1] = (1 / ta) * (i_func(t) - a)
        du[2] = (1 / tm) * (a - m)
        du[3] = 0
        du[4] = (m - c / tc)

        return nothing
    end

    return rhs
end

function represent_on_type(p_derepresented, model_by_type::Type{Model5})
    return ComponentArray(
        tau_a=log(p_derepresented.tau_a),
        tau_m=log(p_derepresented.tau_m),
        tau_c=log(p_derepresented.tau_c),
    )
end

function derepresent(p_repr, model::Model5)
    return ComponentArray(
        tau_a=exp(p_repr.tau_a),
        tau_m=exp(p_repr.tau_m),
        tau_c=exp(p_repr.tau_c),
    )
end

