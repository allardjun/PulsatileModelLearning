
struct MyModel8 <: AbstractClassicalModel
    p_classical_derepresented_ig::ComponentArray{Float64}
    p_derepresented_lowerbounds::ComponentArray{Float64}
    p_derepresented_upperbounds::ComponentArray{Float64}
    u0::Vector{Float64}
    params_repr_ig::ComponentArray{Float64}
    params_derepresented_ig::ComponentArray{Float64}
end

function make_MyModel8()
    p_classical_derepresented_ig = ComponentArray(
        tau_a=5.0f0, tau_m=50.0f0, tau_w=50.0f0, tau_c=100000.0f0, beta_mw=0.01, beta_ma=0.01, nwm=1.9, nam=1.9
    )

    p_derepresented_lowerbounds = ComponentArray(
        tau_a=1.0f0,
        tau_m=1.0f0,
        tau_w=1.0f0,
        # tau_c=1000.0f0 * 1e-2,
        tau_c=1440*10.0, # tauclowerbound 
        beta_mw=1.0 * 1e-3,
        beta_ma=1.0 * 1e-3,
        nwm=0.5,
        nam=0.5,
    )

    p_derepresented_upperbounds = ComponentArray(
        tau_a=5.0f0 * 1e+3,
        tau_m=50.0f0 * 1e+3,
        tau_w=50.0f0 * 1e+3,
        tau_c=1000.0f0 * 1e+3,
        beta_mw=1.0 * 1e+2,
        beta_ma=1.0 * 1e+2,
        nwm=4.0,
        nam=4.0,
    )

    # initial condition
    u0 = [0.0, 0.0, 0.0, 0.0]

    params_repr_ig = ComponentArray( p_classical=represent_on_type(p_classical_derepresented_ig, MyModel8))

    params_derepresented_ig = ComponentArray( p_classical=p_classical_derepresented_ig)

    return MyModel8(
        p_classical_derepresented_ig,
        p_derepresented_lowerbounds,
        p_derepresented_upperbounds,
        u0,
        params_repr_ig,
        params_derepresented_ig,
    )
end

# returns rhs, suitable for inclusion in ODEProblem
function make_rhs(i_func, model::MyModel8)

    # DI-IFFL

    function rhs(du, u, p_all_derepresented, t)
        a, m, w, c = u
        ta, tm, tw, tc, beta_mw, beta_ma, nwm, nam = p_all_derepresented.p_classical

        du[1] = (1 / ta) * (i_func(t) - a)
        du[2] = (1 / tm) * (
            + 1 / (1 + (abs.(beta_mw * w))^nwm) * a 
            - 1 / (1 + (abs.(beta_ma * a))^nam) * m
            )
        du[3] = (1 / tw) * (a - w)
        du[4] = (m - c / tc)

        return nothing
    end

    return rhs
end

function represent_on_type(p_derepresented, model_by_type::Type{MyModel8})
    return ComponentArray(
        tau_a=log(p_derepresented.tau_a),
        tau_m=log(p_derepresented.tau_m),
        tau_w=log(p_derepresented.tau_w),
        tau_c=log(p_derepresented.tau_c),
        beta_mw=log(p_derepresented.beta_mw),
        beta_ma=log(p_derepresented.beta_ma),
        nwm=sqrt(p_derepresented.nwm),
        nam=sqrt(p_derepresented.nam),
    )
end

function derepresent(p_repr, model::MyModel8)
    return ComponentArray(
        tau_a=exp(p_repr.tau_a),
        tau_m=exp(p_repr.tau_m),
        tau_w=exp(p_repr.tau_w),
        tau_c=exp(p_repr.tau_c),
        beta_mw=exp(p_repr.beta_mw),
        beta_ma=exp(p_repr.beta_ma),
        nwm=(p_repr.nwm)^2,
        nam=(p_repr.nam)^2,
    )
end

