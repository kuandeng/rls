function _tri_control_field_dict(control::PsTriRecyControl)
    return Dict{Symbol, Any}(
        name => getfield(control, name) for name in fieldnames(PsTriRecyControl)
    )
end

function _parse_tri_control_value(value_src::AbstractString)
    return Core.eval(@__MODULE__, Meta.parse(strip(value_src)))
end

function _parse_tri_control_arg(arg::AbstractString)
    parts = split(arg, "="; limit = 2)
    length(parts) == 2 || throw(ArgumentError("control override must have the form key=value, got: $arg"))
    key = Symbol(strip(parts[1]))
    value_src = strip(parts[2])
    isempty(value_src) && throw(ArgumentError("control override value cannot be empty for key: $key"))
    return key => _parse_tri_control_value(value_src)
end

function _parse_tri_control_expr(arg::AbstractString)
    expr_src = strip(arg[length("control=")+1:end])
    expr = Meta.parse(expr_src)
    expr isa Expr && expr.head == :call && expr.args[1] == :PsTriRecyControl ||
        throw(ArgumentError("control=... must be a PsTriRecyControl(...) expression"))

    overrides = Pair{Symbol, Any}[]
    for term in expr.args[2:end]
        term isa Expr && term.head == :kw ||
            throw(ArgumentError("PsTriRecyControl(...) only supports keyword arguments in CLI overrides"))
        push!(overrides, term.args[1] => Core.eval(@__MODULE__, term.args[2]))
    end
    return overrides
end

function _merge_tri_control_overrides(
    default_control::PsTriRecyControl,
    overrides::Vector{Pair{Symbol, Any}}
)
    kwargs = _tri_control_field_dict(default_control)
    valid_fields = fieldnames(PsTriRecyControl)

    for (name, value) in overrides
        name in valid_fields || throw(ArgumentError("unknown PsTriRecyControl field: $name"))
        kwargs[name] = value
    end

    return PsTriRecyControl(; kwargs...)
end

function parse_ps_tri_recy_control_args(default_control::PsTriRecyControl)
    overrides = Pair{Symbol, Any}[]
    valid_fields = fieldnames(PsTriRecyControl)

    for arg in ARGS
        if startswith(arg, "control=")
            append!(overrides, _parse_tri_control_expr(arg))
            continue
        end
        occursin("=", arg) || continue
        override = _parse_tri_control_arg(arg)
        override.first in valid_fields || continue
        push!(overrides, override)
    end

    return _merge_tri_control_overrides(default_control, overrides)
end
