function example_output_dir(script_file::AbstractString)
    run_dir = get(ENV, "RLS_RUN_DIR", "")
    output_dir = isempty(run_dir) ? dirname(script_file) : run_dir
    mkpath(output_dir)
    return output_dir
end

function example_output_path(script_file::AbstractString, filename::AbstractString)
    return joinpath(example_output_dir(script_file), filename)
end
