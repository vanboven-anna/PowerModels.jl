using Plots 
using StatsPlots

function plot_var_over_time(var_lst, label, p)
    var_val, val_counter = [], 0
    for (i, lst) in enumerate(var_lst)
        var_val = vcat(var_val, lst)
        val_counter += length(lst)
        # vertical line to indicate swap iteration
        vline!(p, [val_counter], color=:black, linestyle=:dash, label = "")
        vspan!(p, [val_counter-0.1, val_counter+0.1], color=:gray, alpha=0.1, label="")
    end
    # line of variable balues
    plot!(p, 1:length(var_val), var_val, label = label)
    return p
end

function parse_var_vals(x_history, soln_history, mapping_dicts, var, bus_ind, p; gen_ind = nothing, label = nothing)  
    x_val = []
    for (x_hist, sol_hist, md) in zip(x_history, soln_history, mapping_dicts)
        val_iter = nothing
        if var in ["vm", "va"]
            # variable was not solved for
            if md[bus_ind][var] == 0 
                val_iter = [sol_hist["bus"][string(bus_ind)][var] for _ in 1:length(x_hist)]
            else 
                val_iter = [x[md[bus_ind][var]] for x in x_hist]
            end
        else 
            if (md[bus_ind][var] == 0 )|| (md[bus_ind][var] > length(x_hist[1]))
                val_iter = [sol_hist["gen"][string(gen_ind)][var == "q" ? "qg" : "pg"] for _ in 1:length(x_hist)]
            else 
                val_iter = [x[md[bus_ind][var]] for x in x_hist]
            end
        end
        push!(x_val, val_iter)
    end

    return plot_var_over_time(x_val, isnothing(label) ? "$bus_ind" : label, p)
end

function plot_bar_graph_features_scaled(analysis_df, feature_lst)
    analysis_df[!, "run_id"] = 1:nrow(analysis_df)
    # transform into matrix and scale each column
    y_data = Float32.(Matrix(analysis_df[!, feature_lst]))
    col_mins = minimum(y_data, dims=1)
    col_maxs = maximum(y_data, dims=1)
    y_scaled = (y_data .- col_mins) ./ (col_maxs .- col_mins)
    p = groupedbar(
        analysis_df.run_id,             
        y_scaled,                
        bar_position = :dodge,  
        labels = reshape(feature_lst, 1, :),
        xlabel = "Run ID",
        ylabel = "Values",
        legend = :outertopright, 
        xticks = 1:nrow(analysis_df)           
    )
    return p
end

function feature_heat_map(run_info, feature; title = "")
    # transform into matrix and scale each column
    df_wide = unstack(run_info, "run_id", "datapoint", feature, combine = last)
    sort!(df_wide, "run_id")
    y_data = Float32.(Matrix(select(df_wide, Not("run_id"))))
    run_ids = df_wide.run_id
    datapoints = names(select(df_wide, Not(:run_id)))

    h = heatmap(
        datapoints, 
        run_ids, 
        y_data,
        xlabel = "Datapoint",
        ylabel = "Run ID",
        title = title,
        color = :viridis,  
        aspect_ratio = :auto
    )
    
    return h
end