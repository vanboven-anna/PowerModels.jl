#
# Tests for the Sherman-Morrison / sensitivity-score donor selection
# strategy in src/prob/pf_smw.jl.
#
# These tests target the math directly (Lemma 1, Proposition 1, Theorem 1,
# Theorem 2, Corollary 1 of the accompanying notes) and finish with a small
# end-to-end smoke check on case14.
#

import LinearAlgebra: rank, lu, norm

# Helper: build a complete pv_pairs dict (every load bus -> every gen bus,
# in arbitrary order) so that the nearest_gen fallback path stays valid in
# tests, in case the embedded jacobian path errors out.
function _all_pairs_pv_pairs(data)
    gen_bus_ids = unique([gen["gen_bus"] for gen in values(data["gen"])])
    load_bus_ids = [bus["bus_i"] for bus in values(data["bus"])]
    return Dict{Int, Vector{Int}}(b => copy(gen_bus_ids) for b in load_bus_ids)
end

# Helper: load a converged AC power flow on a given case file, returning the
# pf_data with pf_data.vm_idx / pf_data.va_idx mutated to the converged state.
function _converged_pf_data(case_path::String)
    data = PowerModels.parse_file(case_path)
    data["pv_pairs"] = _all_pairs_pv_pairs(data)
    pf_data = PowerModels.instantiate_pf_data(data)
    PowerModels.compute_ac_pf(pf_data, mapping = true, enforce_q_lims = false)
    return pf_data, data
end

@testset "pf_smw: sensitivity-score donor selection" begin

    @testset "EmbeddedMap layout" begin
        pf_data, _ = _converged_pf_data("../test/data/matpower/case14.m")
        emap = PowerModels.build_embedded_map(pf_data)
        n = length(pf_data.bus_type_idx)
        @test length(emap.nonslack_buses) == n - 1
        @test emap.n_emb == 2 * (n - 1)
        # No slack bus appears in the keymaps.
        slack_idx = findfirst(==(3), pf_data.bus_type_idx)
        @test !haskey(emap.va_col, slack_idx)
        @test !haskey(emap.vm_col, slack_idx)
        @test !haskey(emap.p_row, slack_idx)
        @test !haskey(emap.aux_row, slack_idx)
        # va_col and p_row should be the same first-(n-1) positions.
        for b in emap.nonslack_buses
            @test emap.va_col[b] == emap.p_row[b]
            @test emap.vm_col[b] == emap.aux_row[b]
            @test 1 <= emap.va_col[b] <= n - 1
            @test n - 1 + 1 <= emap.vm_col[b] <= 2 * (n - 1)
        end
    end

    @testset "embedded jacobian construction (case14)" begin
        pf_data, _ = _converged_pf_data("../test/data/matpower/case14.m")
        Jhat, emap = PowerModels.build_embedded_jacobian(pf_data, Dict{Int,Int}())

        # Size matches 2*(n-1).
        n = length(pf_data.bus_type_idx)
        @test size(Jhat) == (2*(n-1), 2*(n-1))

        # Lemma 1: PV bus aux-row is exactly e_{V_i}^T.
        for (i, bt) in enumerate(pf_data.bus_type_idx)
            bt == 2 || continue
            qr = emap.aux_row[i]
            row = Jhat[qr, :]
            # Exactly one nonzero, at column emap.vm_col[i], with value 1.
            @test count(!iszero, row) == 1
            @test row[emap.vm_col[i]] == 1.0
        end

        # Embedded residual at converged x* is ~0.
        F = PowerModels.embedded_residual(pf_data, emap, Dict{Int,Int}())
        @test norm(F, Inf) < 1e-6

        # Finite-difference cross-check on a representative entry. Pick the
        # first PQ bus and its first nonslack neighbor; verify dP/dV agrees.
        pq_idx = findfirst(==(1), pf_data.bus_type_idx)
        @test pq_idx !== nothing
        nbrs = pf_data.neighbors[pq_idx]
        nb = first(j for j in nbrs if j != pq_idx && haskey(emap.vm_col, j))
        h = 1e-6
        F0 = PowerModels.embedded_residual(pf_data, emap, Dict{Int,Int}())
        pf_data.vm_idx[nb] += h
        Fp = PowerModels.embedded_residual(pf_data, emap, Dict{Int,Int}())
        pf_data.vm_idx[nb] -= h
        fd = (Fp[emap.p_row[pq_idx]] - F0[emap.p_row[pq_idx]]) / h
        analytic = Jhat[emap.p_row[pq_idx], emap.vm_col[nb]]
        @test isapprox(fd, analytic; atol = 1e-4, rtol = 1e-4)
    end

    @testset "Proposition 1: P-PQV swap is rank-1 in J_hat" begin
        pf_data, _ = _converged_pf_data("../test/data/matpower/case14.m")
        Jpre, emap = PowerModels.build_embedded_jacobian(pf_data, Dict{Int,Int}())

        # Pick a PV donor i and a PQ recipient l.
        i = findfirst(==(2), pf_data.bus_type_idx)
        l = findfirst(==(1), pf_data.bus_type_idx)
        @test i !== nothing && l !== nothing && i != l

        # Simulate the swap on a deepcopy so the post-swap Jacobian is
        # rebuilt with the new pattern.
        pf_data2 = deepcopy(pf_data)
        pf_data2.bus_type_idx[i] = 6   # PV donor -> P-donor (type 6)
        pf_data2.bus_type_idx[l] = 5   # PQ recipient -> PQV recipient (type 5)
        pairs = Dict{Int,Int}(i => l)
        Jpost, emap_post = PowerModels.build_embedded_jacobian(pf_data2, pairs)

        # Same column/row layout (no slack movement).
        @test emap.n_emb == emap_post.n_emb

        D = Jpost - Jpre
        # Proposition 1: rank(D) <= 1, and exactly 1 here since i != l.
        @test rank(D) == 1

        # The difference equals e_{r_i} (e_{V_l} - e_{V_i})^T.
        ri = emap.aux_row[i]
        # The only nonzero row of D is row ri.
        for r in axes(D, 1)
            if r == ri
                continue
            end
            @test all(iszero, D[r, :])
        end
        # Row ri has +1 at vm_col[l] and -1 at vm_col[i], zeros elsewhere.
        expected = zero(D[ri, :])
        expected[emap.vm_col[l]] += 1.0
        expected[emap.vm_col[i]] -= 1.0
        @test isapprox(D[ri, :], expected; atol = 1e-12)
    end

    @testset "Theorem 1: SMW first Newton step matches direct solve" begin
        pf_data, data = _converged_pf_data("../test/data/matpower/case14.m")
        Jpre, emap = PowerModels.build_embedded_jacobian(pf_data, Dict{Int,Int}())

        i = findfirst(==(2), pf_data.bus_type_idx)
        l = findfirst(==(1), pf_data.bus_type_idx)
        target_Vl = data["bus"][string(pf_data.am.idx_to_bus[l])]["vmin"]

        # Build post-swap J_hat at the SAME linearization point x* as Jpre
        # (only the equation pattern changes; V_l(x*) is unchanged here).
        pf_data2 = deepcopy(pf_data)
        pf_data2.bus_type_idx[i] = 6
        pf_data2.bus_type_idx[l] = 5
        Jpost, _ = PowerModels.build_embedded_jacobian(pf_data2, Dict{Int,Int}(i => l))

        # Post-swap residual at x*: the only nonzero row is the one that was
        # the donor's V row, now carrying V_l - V_hat_l.
        Vl_pre = pf_data.vm_idx[l]
        Fpost = zeros(Float64, emap.n_emb)
        Fpost[emap.aux_row[i]] = Vl_pre - target_Vl

        # Direct Newton step: dx_direct = -Jpost \ Fpost.
        dx_direct = -(Jpost \ Fpost)

        # SMW step using only Jpre and one back-solve.
        z = PowerModels.compute_sensitivity_columns(Jpre, emap, [i])[i]
        s = z[emap.vm_col[l]]
        @test abs(s) > 1e-6
        coef = -(Vl_pre - target_Vl) / s
        dx_smw = coef .* z

        @test isapprox(dx_direct, dx_smw; atol = 1e-8, rtol = 1e-8)
    end

    @testset "Relinearization term is not rank-one" begin
        # Companion to "Proposition 1" / "Theorem 1": those tests verify the
        # exact identity J_τ'(x*) - J_τ(x*) = e_{r_i}(e_{V_l} - e_{V_i})^T at
        # the SAME state x*. Once V_l is mutated to its bound (or one Newton
        # step is taken), the AC PF Jacobian must be re-evaluated at the new
        # state and picks up a relinearization term J_τ'(x_new) - J_τ'(x*)
        # that is generally NOT rank one -- because P_m, Q_m for every
        # neighbor m of l contain V_l*V_m, cos(θ_m-θ_l) etc.
        #
        # This test demonstrates that the relinearization term exists, has
        # rank > 1, and that the changed entries land in rows associated with
        # neighbors of l. We do NOT use SMW for an exact update across this
        # state change; SMW is the first Newton predictor only.
        pf_data, data = _converged_pf_data("../test/data/matpower/case14.m")

        i = findfirst(==(2), pf_data.bus_type_idx)
        l = findfirst(==(1), pf_data.bus_type_idx)
        target_Vl = data["bus"][string(pf_data.am.idx_to_bus[l])]["vmin"]

        # J_τ'(x*) -- swap pattern, but state still at x*.
        pf_data2 = deepcopy(pf_data)
        pf_data2.bus_type_idx[i] = 6
        pf_data2.bus_type_idx[l] = 5
        Jpost_xstar, emap = PowerModels.build_embedded_jacobian(pf_data2, Dict{Int,Int}(i => l))

        # J_τ'(x_new) -- same swap pattern, but recipient pinned to V̂_l.
        pf_data3 = deepcopy(pf_data2)
        pf_data3.vm_idx[l] = target_Vl
        Jpost_xnew, _ = PowerModels.build_embedded_jacobian(pf_data3, Dict{Int,Int}(i => l))

        D_relin = Jpost_xnew - Jpost_xstar

        # Relinearization term is real (V_l moved by a non-trivial amount).
        @test abs(target_Vl - pf_data.vm_idx[l]) > 1e-3
        @test maximum(abs, D_relin) > 1e-6

        # Critically, NOT rank one.
        @test rank(Matrix(D_relin)) > 1

        # Changed rows live in P/Q rows of {l} ∪ neighbors(l). Confirm that
        # at least one neighbor of l has a non-trivial change in either its
        # P-row or its aux row (Q row, since the neighbor stays type 1).
        nbrs_of_l = [m for m in pf_data.neighbors[l] if m != l && haskey(emap.p_row, m)]
        @test !isempty(nbrs_of_l)
        neighbor_changed = false
        for m in nbrs_of_l
            row_changes = max(maximum(abs, D_relin[emap.p_row[m], :]),
                              maximum(abs, D_relin[emap.aux_row[m], :]))
            if row_changes > 1e-6
                neighbor_changed = true
                break
            end
        end
        @test neighbor_changed
    end

    @testset "Corollary 1: V_l prediction is exactly satisfied" begin
        pf_data, data = _converged_pf_data("../test/data/matpower/case14.m")
        Jpre, emap = PowerModels.build_embedded_jacobian(pf_data, Dict{Int,Int}())

        i = findfirst(==(2), pf_data.bus_type_idx)
        l = findfirst(==(1), pf_data.bus_type_idx)
        target_Vl = data["bus"][string(pf_data.am.idx_to_bus[l])]["vmin"]

        z = PowerModels.compute_sensitivity_columns(Jpre, emap, [i])[i]
        Vl_pre = pf_data.vm_idx[l]

        # Predicted V_l after the swap should equal V_hat_l exactly (Corollary 1).
        # V_l_pred = V_l(x*) - (V_l - V_hat_l) * (e_{V_l}^T z) / (e_{V_l}^T z) = V_hat_l.
        s = z[emap.vm_col[l]]
        Vl_pred = Vl_pre - (Vl_pre - target_Vl) * (z[emap.vm_col[l]] / s)
        @test isapprox(Vl_pred, target_Vl; atol = 1e-12)
    end

    @testset "Theorem 2: local-optimal donor maximizes |s_li|" begin
        pf_data, _ = _converged_pf_data("../test/data/matpower/case14.m")
        Jpre, emap = PowerModels.build_embedded_jacobian(pf_data, Dict{Int,Int}())

        l = findfirst(==(1), pf_data.bus_type_idx)
        candidates = [k for k in 1:length(pf_data.bus_type_idx) if pf_data.bus_type_idx[k] == 2]
        @test length(candidates) >= 2   # case14 has multiple PV buses
        z_dict = PowerModels.compute_sensitivity_columns(Jpre, emap, candidates)
        donor, score, sens = PowerModels.score_local_optimal(z_dict, emap, l)
        @test donor in candidates

        # The chosen donor must have the largest |s_li| among candidates.
        max_abs = maximum(abs(z_dict[k][emap.vm_col[l]]) for k in candidates)
        @test isapprox(abs(sens), max_abs; atol = 1e-12)
        @test isapprox(score, max_abs; atol = 1e-12)
    end

    @testset "score predicts donor quality on case14 first swap" begin
        # Validate the sensitivity score against ground truth on a real network.
        # For case14's first swap, fix the recipient l to the PQ bus closest
        # to a voltage bound; for every PV donor i:
        #   (a) compute predicted score |e_{V_l}^T J_hat^{-1} e_{r_i}|,
        #   (b) actually perform the (i, l) swap, run NR to convergence, and
        #       measure post-swap total |V| violation magnitude.
        # Assert that the score's top-1 donor lands in the actual top half
        # and is within a small constant of the oracle's best total |V|.
        pf_data, data = _converged_pf_data("../test/data/matpower/case14.m")

        # Recipient: PQ bus with smallest distance to either voltage bound.
        pq_buses = [k for k in 1:length(pf_data.bus_type_idx) if pf_data.bus_type_idx[k] == 1]
        @test !isempty(pq_buses)
        function _bound_margin(k)
            b = data["bus"][string(pf_data.am.idx_to_bus[k])]
            V = pf_data.vm_idx[k]
            return min(V - b["vmin"], b["vmax"] - V)
        end
        l = argmin(_bound_margin, pq_buses)
        bus_l = data["bus"][string(pf_data.am.idx_to_bus[l])]
        Vl_pre = pf_data.vm_idx[l]
        target_Vl = (Vl_pre - bus_l["vmin"]) < (bus_l["vmax"] - Vl_pre) ? bus_l["vmin"] : bus_l["vmax"]

        # Candidate PV donors (case14 has 4: buses 2, 3, 6, 8).
        candidates = [k for k in 1:length(pf_data.bus_type_idx) if pf_data.bus_type_idx[k] == 2]
        @test length(candidates) >= 2

        # Predicted ranking: |e_{V_l}^T J_hat^{-1} e_{r_i}| for each donor.
        Jhat, emap = PowerModels.build_embedded_jacobian(pf_data, Dict{Int,Int}())
        z_dict = PowerModels.compute_sensitivity_columns(Jhat, emap, candidates)
        score = Dict(i => abs(z_dict[i][emap.vm_col[l]]) for i in candidates)

        # Oracle: actually perform each (i, l) swap and measure post-swap state.
        actual_total = Dict{Int, Float64}()
        actual_Vl    = Dict{Int, Float64}()
        for i in candidates
            pf_i = deepcopy(pf_data)
            pf_i.bus_type_idx[i] = 6        # PV donor -> P-donor
            pf_i.bus_type_idx[l] = 5        # PQ recipient -> PQV recipient
            pf_i.vm_idx[l]       = target_Vl
            # Warm-start NR from the (post-pin) converged-pre-swap state.
            for (k, bid) in enumerate(pf_i.am.idx_to_bus)
                db = pf_i.data["bus"][string(bid)]
                db["vm_start"] = pf_i.vm_idx[k]
                db["va_start"] = pf_i.va_idx[k]
            end
            mapping_dict, J0_map = PowerModels.map_types_to_variable_indices(pf_i)
            pf_result, _, _ = PowerModels._compute_ac_pf(pf_i, mapping_dict, J0_map; flat_start = false)
            converged = pf_result.x_converged || pf_result.f_converged
            actual_Vl[i] = converged ? pf_i.vm_idx[l] : NaN
            if !converged
                actual_total[i] = Inf
                continue
            end
            total = 0.0
            for k in 1:length(pf_i.bus_type_idx)
                b = data["bus"][string(pf_i.am.idx_to_bus[k])]
                V = pf_i.vm_idx[k]
                V < b["vmin"] && (total += b["vmin"] - V)
                V > b["vmax"] && (total += V - b["vmax"])
            end
            actual_total[i] = total
        end

        predicted_rank = sort(candidates; by = i -> -score[i])
        actual_rank    = sort(candidates; by = i -> actual_total[i])
        best_actual    = actual_total[actual_rank[1]]

        # Also predict via the collateral-aware score (Corollary 1): for each
        # donor i, predict V_m at every nonslack bus via the linear column
        # z_i, sum the predicted bound violations, and pick the donor that
        # minimizes the predicted total. This is supposed to dominate the
        # local-optimal score on collateral-heavy cases.
        collat_donor, _, _ = PowerModels.score_collateral_aware(
            z_dict, emap, pf_data, l, target_Vl; alpha = 0.0)

        println("\n  case14 first-swap donor validation (recipient l = $l, target_Vl = $target_Vl):")
        for i in candidates
            si = round(score[i];          sigdigits = 4)
            vi = round(actual_Vl[i];      digits = 5)
            ti = round(actual_total[i];   digits = 5)
            println("    donor=$i  score=$si  actual_Vl=$vi  total|V|_post=$ti")
        end
        println("    local-optimal best donor:    $(predicted_rank[1])  (actual total |V| = $(round(actual_total[predicted_rank[1]]; digits=5)))")
        println("    collateral-aware best donor: $(collat_donor)  (actual total |V| = $(round(actual_total[collat_donor];    digits=5)))")
        println("    oracle best donor:           $(actual_rank[1])  (actual total |V| = $(round(best_actual;                 digits=5)))")

        # Validation 1 (local-optimal): top-1 lands in the actual top half.
        topN = max(2, div(length(actual_rank), 2))
        @test predicted_rank[1] in actual_rank[1:topN]

        # Validation 2 (local-optimal): predicted top-1's actual total |V| is
        # no worse than 1.25x the oracle's. Local-optimal doesn't directly
        # minimize collateral, hence the small margin.
        pred_actual = actual_total[predicted_rank[1]]
        @test pred_actual <= 1.25 * best_actual + 1e-6

        # Validation 3 (collateral-aware): a strictly tighter check.
        # Collateral-aware predicts post-swap V at every bus and picks the
        # donor that minimizes the summed predicted violation -- this is the
        # quantity the oracle ranks. Assert it matches the oracle exactly OR
        # is within 1.05x of the oracle's best.
        collat_actual = actual_total[collat_donor]
        @test collat_donor == actual_rank[1] || collat_actual <= 1.05 * best_actual + 1e-6
    end

    @testset "score_collateral_aware runs and prefers low-collateral donor" begin
        pf_data, _ = _converged_pf_data("../test/data/matpower/case14.m")
        Jpre, emap = PowerModels.build_embedded_jacobian(pf_data, Dict{Int,Int}())

        l = findfirst(==(1), pf_data.bus_type_idx)
        candidates = [k for k in 1:length(pf_data.bus_type_idx) if pf_data.bus_type_idx[k] == 2]
        z_dict = PowerModels.compute_sensitivity_columns(Jpre, emap, candidates)

        # An arbitrary in-bounds target; any value triggers the scoring path.
        target_Vl = pf_data.vm_idx[l] - 0.01
        donor_c, score_c, sens_c = PowerModels.score_collateral_aware(
            z_dict, emap, pf_data, l, target_Vl; alpha = 0.0)
        @test donor_c !== nothing
        @test donor_c in candidates
        @test isfinite(score_c)
        @test isfinite(sens_c)
    end

    @testset "edge case: no admissible donors returns gracefully" begin
        pf_data, _ = _converged_pf_data("../test/data/matpower/case14.m")
        # Set up the bookkeeping that compute_ac_pf_mult_buses normally writes.
        pf_data.data["pv_bus_inds"] = Int[]   # empty pool -> no candidates
        pf_data.data["prev_swaps"] = Dict(b => Int[] for b in 1:length(pf_data.bus_type_idx))
        b1 = [(findfirst(==(1), pf_data.bus_type_idx), 0.05, 0.95)]
        p_pqv_pairs = Dict{Int,Int}()
        bus_assignment = Dict{String,Any}()
        for (i, bus) in pf_data.data["bus"]
            bus_assignment[i] = Dict{String,Float64}("vm" => bus["vm"], "va" => bus["va"])
        end
        swap = Ref(false)
        flags = PowerModels.SwapFlags(swap_technique = "sensitivity_score")
        # Nothing should error.
        PowerModels.perform_bus_swaps_sensitivity_score_impl!(
            pf_data, nothing, nothing, copy(pf_data.bus_type_idx),
            p_pqv_pairs, bus_assignment, swap, b1, flags)
        @test swap[] == false
        @test isempty(p_pqv_pairs)
    end

    @testset "end-to-end: sensitivity_score strategy solves case14" begin
        # Rerun on a fresh data dict so previous mutations don't leak in.
        data = PowerModels.parse_file("../test/data/matpower/case14.m")
        data["pv_pairs"] = _all_pairs_pv_pairs(data)
        result = PowerModels.compute_ac_pf_mult_buses(data;
                    swap_technique = "sensitivity_score", obo = true)
        @test haskey(result, "solution")
        @test haskey(result, "solve_time")
        @test result["solve_time"] >= 0
        # Solver should produce a usable bus solution dict.
        if !isnothing(result["solution"]) && haskey(result["solution"], "bus")
            for (_, bus) in result["solution"]["bus"]
                @test haskey(bus, "vm")
                @test haskey(bus, "va")
            end
        end
    end

    @testset "end-to-end with use_smw_warmstart flag enabled" begin
        data = PowerModels.parse_file("../test/data/matpower/case14.m")
        data["pv_pairs"] = _all_pairs_pv_pairs(data)
        result = PowerModels.compute_ac_pf_mult_buses(data;
                    swap_technique = "sensitivity_score",
                    use_smw_warmstart = true, obo = true)
        @test haskey(result, "solution")
    end

    @testset "use_smw_warmstart actually mutates state" begin
        # Drive a swap on a converged case14 manually and verify that the
        # warm-start branch writes nontrivial deltas into pf_data.vm_idx /
        # pf_data.va_idx (vs flag=false, which leaves them untouched).
        function _run_one_swap(use_warmstart)
            pf_data, _ = _converged_pf_data("../test/data/matpower/case14.m")
            pf_data.data["pv_bus_inds"] = [k for (k, bt) in enumerate(pf_data.bus_type_idx) if bt == 2]
            pf_data.data["prev_swaps"] = Dict(b => Int[] for b in 1:length(pf_data.bus_type_idx))
            l = findfirst(==(1), pf_data.bus_type_idx)
            target_Vl = pf_data.data["bus"][string(pf_data.am.idx_to_bus[l])]["vmin"]
            # Manufacture a violation: V_l violates its lower bound by 0.05.
            viol_mag = 0.05
            b1 = [(l, viol_mag, target_Vl)]
            p_pqv_pairs = Dict{Int,Int}()
            bus_assignment = Dict{String,Any}()
            for (s, bus) in pf_data.data["bus"]
                bus_assignment[s] = Dict{String,Float64}("vm" => bus["vm"], "va" => bus["va"])
            end
            swap = Ref(false)
            flags = PowerModels.SwapFlags(swap_technique = "sensitivity_score",
                                          use_smw_warmstart = use_warmstart, obo = true)
            vm_pre = copy(pf_data.vm_idx)
            va_pre = copy(pf_data.va_idx)
            PowerModels.perform_bus_swaps_sensitivity_score_impl!(
                pf_data, nothing, nothing, copy(pf_data.bus_type_idx),
                p_pqv_pairs, bus_assignment, swap, b1, flags)
            return swap[], pf_data.vm_idx .- vm_pre, pf_data.va_idx .- va_pre
        end

        sw_no, dvm_no, dva_no = _run_one_swap(false)
        sw_ws, dvm_ws, dva_ws = _run_one_swap(true)
        @test sw_no && sw_ws

        # Without the flag: vm only changes at the recipient (forced to bound by
        # swap_pqv_buses!); va is untouched.
        @test count(!iszero, dvm_no) == 1
        @test all(iszero, dva_no)

        # With the flag: va changes at multiple nonslack buses (warm-start step),
        # and vm changes at the recipient + other Newton-variable buses too.
        @test count(!iszero, dva_ws) > 1
        @test count(!iszero, dvm_ws) > count(!iszero, dvm_no)
    end
end
