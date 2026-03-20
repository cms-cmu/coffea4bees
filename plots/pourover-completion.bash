# PourOver bash completion + alias
#
# Source from the barista root directory (add to ~/.bashrc):
#   source /path/to/barista/coffea4bees/plots/pourover-completion.bash
#
# Then use:
#   pourover file.coffea -m meta.yml --new --label ul18-sb
#   pourover --load <TAB>

alias pourover='python coffea4bees/plots/pourOver.py'

_pourover_labels() {
    local registry="${1:-.pourover_archives/pourover_registry.json}"
    [[ -f "$registry" ]] || return
    python3 -c "
import json
try:
    seen, out = set(), []
    for e in reversed(json.load(open('$registry'))):
        l = e.get('label', '')
        if l and l not in seen:
            out.append(l)
            seen.add(l)
    print('\n'.join(out))
except Exception:
    pass
" 2>/dev/null
}

_pourover() {
    local cur prev words cword
    if declare -f _init_completion > /dev/null 2>&1; then
        _init_completion || return
    else
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        words=("${COMP_WORDS[@]}")
        cword=$COMP_CWORD
    fi

    # Find --registry value if set earlier on the line
    local registry=".pourover_archives/pourover_registry.json"
    local i
    for ((i=1; i < cword; i++)); do
        if [[ "${words[i]}" == "--registry" ]]; then
            registry="${words[i+1]}"
        fi
    done

    case "$prev" in
        --load|--rename|--delete)
            local labels
            labels=$(_pourover_labels "$registry")
            COMPREPLY=($(compgen -W "$labels" -- "$cur"))
            return
            ;;
        --registry|--output-dir)
            COMPREPLY=($(compgen -f -- "$cur"))
            return
            ;;
        -m|--metadata)
            COMPREPLY=($(compgen -f -X '!*.@(yml|yaml)' -- "$cur")
                       $(compgen -d -- "$cur"))
            return
            ;;
        --port|-j|--jobs|--label)
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "
            --new --load --list --rename --delete --registry --label
            --port --pregallery --reuse-gallery
            -j --jobs --output-dir -m --metadata
        " -- "$cur"))
        return
    fi

    # Default: complete .coffea files and directories
    COMPREPLY=($(compgen -f -X '!*.coffea' -- "$cur")
               $(compgen -d -- "$cur"))
}

complete -F _pourover pourover
