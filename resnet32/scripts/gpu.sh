# Sourced by the submit scripts. gpu_options <GB> prints the sbatch options for the smallest A100 (MIG slice
# or whole card) with at least <GB> GB, read from sinfo; GRES=gpu:<type>:1 skips the lookup.
gpu_options() {
    if [[ -n "${GRES:-}" ]]; then
        echo "--gres=$GRES"
        return 0
    fi
    local opts
    opts=$(sinfo -h -o '%P|%G' 2>/dev/null | awk -F'|' -v need="$1" '
        {
            part = $1
            dflt = sub(/\*$/, "", part)
            gres = $2
            gsub(/\([^)]*\)/, "", gres)
            n = split(gres, items, ",")
            for (i = 1; i <= n; i++) {
                if (split(items[i], f, ":") < 3 || f[1] != "gpu") continue
                t = tolower(f[2])
                if (match(t, /[0-9]+g\.[0-9]+gb/)) { mem = substr(t, RSTART, RLENGTH); full = 0 }
                else if (t ~ /a100|ampere/) { mem = match(t, /[0-9]+gb/) ? substr(t, RSTART, RLENGTH) : "40gb"; full = 1 }
                else continue
                sub(/^[0-9]+g\./, "", mem)
                mem += 0
                if (mem < need) continue
                score = 4 * mem + 2 * full + !dflt
                if (best == "" || score < best_score) { best = f[2]; best_part = dflt ? "" : part; best_score = score }
            }
        }
        END { if (best != "") print "--gres=gpu:" best ":1" (best_part == "" ? "" : " --partition=" best_part) }')
    if [[ -z "$opts" ]]; then
        echo "No A100 with at least $1 GB found by sinfo; set GRES=gpu:<type>:1. GPUs per partition:" >&2
        sinfo -o '%P %G' >&2 || true
        return 1
    fi
    echo "$opts"
}
