"""Caricamento e composizione delle configurazioni degli esperimenti.

Una config e' un dizionario annidato caricato da YAML. Supporta:
- ereditarieta': un file puo' dichiarare `base: altro.yaml` e una sezione
  `override:` che ne modifica solo alcuni campi (deep-merge);
- sweep: una sezione `sweep:` con liste di valori per chiavi "dotted"
  (es. "optim.lr"); espansa nel prodotto cartesiano -> una config per combinazione.

Dipendenze intenzionalmente leggere: solo stdlib (os, copy, itertools) + PyYAML.
Questo modulo appartiene al layer di orchestrazione: viene chiamato da
experiment.py e run.py per ottenere le config concrete su cui girare i modelli.
"""
import os
import copy
import itertools
import yaml


# ---------------------------------------------------------------------------
# Funzioni helper (interne)
# ---------------------------------------------------------------------------

def _read_yaml(path):
    """Legge un file YAML e restituisce il dizionario (o {} se vuoto)."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base, override):
    """Ritorna una copia di *base* con i campi di *override* applicati ricorsivamente.

    Regola: se un valore in override E' un dict e anche il corrispondente
    valore in base E' un dict, si fa il merge ricorsivo (cosi' si possono
    sovrascrivere singoli sotto-campi senza cancellare i fratelli).
    In tutti gli altri casi il valore di override vince direttamente.
    """
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            # Entrambi dizionari: merge ricorsivo per preservare i sotto-campi
            # di base che non compaiono in override.
            out[k] = _deep_merge(out[k], v)
        else:
            # Valore scalare, lista, o override introduce una nuova chiave.
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

def set_dotted(d, dotted_key, value):
    """Imposta ``d[a][b][c] = value`` data la chiave "a.b.c".

    Crea automaticamente i dizionari intermedi mancanti (comportamento
    analogo a ``setdefault`` ricorsivo). Utile sia per applicare gli
    override della sezione *sweep* sia per unit-test isolati.

    Args:
        d (dict): dizionario target, modificato in-place.
        dotted_key (str): chiave in notazione dotted, es. "optim.lr".
        value: valore da assegnare alla foglia.
    """
    keys = dotted_key.split(".")
    cur = d
    # Naviga fino al penultimo livello, creando i dict mancanti.
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    # Assegna alla chiave foglia.
    cur[keys[-1]] = value


def load_config(path):
    """Carica una config YAML risolvendo ereditarieta' (``base``) e ``override``.

    Se il file contiene la chiave ``base``, questa e' interpretata come path
    relativo alla directory del file corrente. Il file base viene caricato
    prima, poi la sezione ``override:`` del file corrente viene applicata
    con un deep-merge, cosi' i campi non menzionati in override vengono
    ereditati intatti dal file base.

    Se non c'e' ``base``, il file viene restituito as-is.

    Args:
        path (str): percorso assoluto o relativo al file YAML dell'esperimento.

    Returns:
        dict: config risolta, pronta per essere passata a expand_sweep.
    """
    raw = _read_yaml(path)

    if "base" in raw:
        # Risolve il path del file base rispetto alla directory del file corrente,
        # cosi' i file possono essere spostati insieme senza rompere i riferimenti.
        base_path = os.path.join(os.path.dirname(path), raw["base"])
        base = _read_yaml(base_path)
        # Deep-merge: base fornisce i default, override li sovrascrive selettivamente.
        cfg = _deep_merge(base, raw.get("override", {}))
        # Preserva la sezione sweep (non fa parte dell'override, e' un livello
        # meta della config che descrive quali combinazioni di iperparametri
        # espandere; non deve essere ereditata dal base ne' persa nel merge).
        if "sweep" in raw:
            cfg["sweep"] = raw["sweep"]
    else:
        cfg = raw

    return cfg


def expand_sweep(cfg):
    """Espande la sezione ``sweep`` nel prodotto cartesiano di config concrete.

    La sezione ``sweep`` e' un dict ``{dotted_key: [v1, v2, ...]}``.
    Ogni combinazione (prodotto cartesiano di tutte le liste) genera una
    config indipendente:
    - la chiave ``sweep`` viene rimossa dalla config concreta;
    - i valori vengono scritti nelle posizioni corrette tramite set_dotted;
    - il nome dell'esperimento viene decorato con un suffisso che descrive
      la combinazione (es. "base__lr0.0005__epochs20") per unicita'.

    Se ``sweep`` e' assente, restituisce una lista con la sola config originale
    (nessuna modifica), compatibile con i chiamanti che si aspettano sempre
    una lista.

    Args:
        cfg (dict): config caricata da load_config, eventualmente con ``sweep``.

    Returns:
        list[dict]: lista di config concrete, una per combinazione di sweep.
    """
    sweep = cfg.get("sweep")

    # Caso senza sweep: restituisce lista unitaria con copia della config.
    if not sweep:
        return [copy.deepcopy(cfg)]

    # Estrae chiavi e liste di valori mantenendo l'ordine (Python 3.7+ dict).
    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys]

    runs = []
    for combo in itertools.product(*value_lists):
        # Copia profonda per isolare ogni run dalle altre.
        run_cfg = copy.deepcopy(cfg)
        # Rimuove la sezione sweep: la run concreta non la deve contenere.
        run_cfg.pop("sweep", None)

        # Applica i valori della combinazione corrente e costruisce il suffisso
        # del nome usando solo la parte finale della chiave dotted (es. "lr").
        suffix_parts = []
        for k, v in zip(keys, combo):
            set_dotted(run_cfg, k, v)
            suffix_parts.append(f"{k.split('.')[-1]}{v}")

        # Deriva un nome univoco per la run aggiungendo il suffisso al nome base.
        base_name = cfg.get("experiment", {}).get("name", "run")
        run_cfg.setdefault("experiment", {})["name"] = (
            base_name + "__" + "__".join(suffix_parts)
        )
        runs.append(run_cfg)

    return runs
