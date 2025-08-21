struct ModelF6 <: AbstractFlexiModel
    p_classical_derepresented_ig::ComponentArray{Float64}
    p_derepresented_lowerbounds::ComponentArray{Float64}
    p_derepresented_upperbounds::ComponentArray{Float64}
    u0::Vector{Float64}
    params_repr_ig::ComponentArray{Float64}
    params_derepresented_ig::ComponentArray{Float64}
end

function make_ModelF6(; flexi_dofs=20)
    p_classical_derepresented_ig = ComponentArray(
        tau_a=5.0,
        tau_m=50.0,
        # tau_w = 50,
        tau_c=1000.0,
        # beta_mw = 0.01,
        # beta_ma=0.01,
        # nwm = 1.9,
        # nam=1.9,
    )

    p_derepresented_lowerbounds = ComponentArray(
        tau_a=1.0,
        tau_m=1.0,
        # tau_w = 1,
        tau_c=1000.0 * 1e-2,
        # beta_mw = 1.0*1e-3,
        # beta_ma=1.0 * 1e-3,
        # nwm = 0.5,
        # nam=0.5,
    )

    p_derepresented_upperbounds = ComponentArray(
        tau_a=5.0 * 1e+3,
        tau_m=50.0 * 1e+3,
        # tau_w = 50*1e+3,
        tau_c=1000.0 * 1e+3,
        # beta_mw = 1.0*1e+2,
        # beta_ma=1.0 * 1e+2,
        # nwm = 4.0,
        # nam=4.0,
    )

    # initial condition
    u0 = [0.0, 0.0, 0.0, 0.0]

    params_repr_ig = ComponentArray(
        p_classical=represent_on_type(p_classical_derepresented_ig, ModelF6),
        flex1_params=FlexiFunctions.generate_flexi_ig(flexi_dofs),
        flex2_params=FlexiFunctions.generate_flexi_ig(flexi_dofs),
    )

    params_derepresented_ig = ComponentArray(
        p_classical=deepcopy(p_classical_derepresented_ig),
        flex1_params=FlexiFunctions.generate_flexi_ig(flexi_dofs),
        flex2_params=FlexiFunctions.generate_flexi_ig(flexi_dofs),
    )

    return ModelF6(
        p_classical_derepresented_ig,
        p_derepresented_lowerbounds,
        p_derepresented_upperbounds,
        u0,
        params_repr_ig,
        params_derepresented_ig,
    )
end

# returns rhs, suitable for inclusion in ODEProblem
function make_rhs(i_func, model::ModelF6)

    # linear architecture with decay-inhibition and flexi function

    function rhs(du, u, p_all_derepresented, t)
        a, m, w, c = u
        ta, tm, tc = p_all_derepresented.p_classical

        du[1] = (1 / ta) * (i_func(t) - a)
        du[2] = (1 / tm) * (a - 
            1 / (1 + abs(FlexiFunctions.evaluate_decompress(abs(a), p_all_derepresented.flex1_params))) * m)
        du[3] = 0
        du[4] = (m - c / tc)

        return nothing
    end

    return rhs
end

function represent_on_type(p_derepresented, model_by_type::Type{ModelF6})
    return ComponentArray(
        tau_a=log(p_derepresented.tau_a),
        tau_m=log(p_derepresented.tau_m),
        # tau_w = log(p_derepresented.tau_w),
        tau_c=log(p_derepresented.tau_c),
        # beta_mw = log(p_derepresented.beta_mw),
        # beta_ma=log(p_derepresented.beta_ma),
        # nwm = sqrt(p_derepresented.nwm),
        # nam=sqrt(p_derepresented.nam),
    )
end

function derepresent(p_repr, model::ModelF6)
    return ComponentArray(
        tau_a=exp(p_repr.tau_a),
        tau_m=exp(p_repr.tau_m),
        # tau_w = exp(p_repr.tau_w),
        tau_c=exp(p_repr.tau_c),
        # beta_mw = exp(p_repr.beta_mw),
        # beta_ma=exp(p_repr.beta_ma),
        # nwm = (p_repr.nwm)^2,
        # nam=(p_repr.nam)^2,
    )
end