# MDLM Sampling Study — JAX

Étude expérimentale au niveau caractère : **MDLM/SUBS, Transformer AR avec KV-cache, MLM, unigramme et trigramme**, entraînés et évalués sur un sous-ensemble de Tiny Shakespeare.

- [Rapport avec figures et résultats mesurés](results/study/REPORT.html)
- [Rapport Markdown](results/study/REPORT.md)
- [Audit des références indépendantes et limites de comparaison](results/study/REPORT.html#independent-reproduction)
- [Effet mesuré des étapes : générations répétées et incertitude](results/study/REPORT.html#sampling-statistics)
- [Comparaison des modèles](results/study/model_metrics.csv)
- [Latence et mémoire](results/study/timings.csv)
- [Ablations des samplers](results/study/sampling_ablations.csv)
- [Tests exécutés](results/tests.json)
- [Preuve de fonctionnement sur une distribution connue](results/proof/PROOF.html)

## Preuve contrôlée, calculée exactement

La section 11 du rapport distingue les résultats indépendants publiés (VADD), le candidat logiciel MDLM standard de MDM-Prime et nos mesures locales. Les dépôts tiers n’ont pas été exécutés. Aucune chaîne de validation complète des benchmarks originaux jusqu’à notre sous-ensemble Tiny Shakespeare n’est établie. Le budget CPU motive une étude réduite, pas une revendication de reproduction du papier. Les révisions et limites vérifiées sont conservées dans [le relevé d’audit](references/independent_audit.json).

`proof.py` entraîne le même mécanisme MDLM depuis zéro sur des mots binaires de huit bits : la seconde moitié doit recopier la première, avec 16 modes équiprobables. Une énumération indépendante de toute la chaîne générative calcule la probabilité de validité, la distance à la vraie distribution et la **NLL exacte**. Des tirages du sampler JAX sont comparés à ce calcul. La validité seule ne suffit pas : les critères vérifient aussi la diversité, pour détecter un modèle qui répète toujours le même mot.

```sh
JAX_PLATFORMS=cpu python3 proof.py run
JAX_PLATFORMS=cpu python3 check.py --output results/proof/tests.json
```

Les critères sont fixés dans `configs/proof.json`, avant les trois entraînements. Cette preuve contrôlée ne constitue pas une reproduction du papier ni une validation de la qualité linguistique sur Shakespeare.

## Exécuter

L’étude statistique complémentaire réutilise les trois checkpoints MDLM et génère de nouveaux textes, sans réentraînement : `JAX_PLATFORMS=cpu python3 sampling_study.py`. Son protocole et les générations sont sauvegardés dans `results/study/sampling_statistics/`. Il s’agit d’un suivi exploratoire du balayage initial, avec incertitude de génération conditionnelle aux checkpoints ; pas d’une reproduction des scores du papier.

Python 3.13 utilisé pour l'exécution livrée. JAX pur : pas de PyTorch, Flax ni Optax. Le corpus public est inclus pour permettre la reproduction hors ligne une fois les dépendances installées.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
JAX_PLATFORMS=cpu .venv/bin/python check.py
JAX_PLATFORMS=cpu .venv/bin/python run.py all --config configs/smoke.json --output results/smoke
JAX_PLATFORMS=cpu .venv/bin/python run.py all --config configs/study.json --output results/study
```

`study.json` est la configuration principale retenue après exploration sur validation. Les résultats déjà livrés contiennent les checkpoints : `train` saute les entraînements terminés et reprend ceux qui sont interrompus. Pour un nouvel entraînement indépendant, utiliser un nouveau dossier de sortie, par exemple `--output results/reproduction`.

Les étapes peuvent être séparées :

```sh
JAX_PLATFORMS=cpu python3 run.py train --config configs/study.json --output results/study
JAX_PLATFORMS=cpu python3 run.py evaluate --config configs/study.json --output results/study
JAX_PLATFORMS=cpu python3 run.py benchmark --config configs/study.json --output results/study
JAX_PLATFORMS=cpu python3 run.py report --config configs/study.json --output results/study
```

Pour entraîner seulement certains modèles, ajouter par exemple `--runs mdlm_s0 ar_s0`. La production du rapport complet exige tous les modèles de la configuration. Une modification de configuration dans un dossier possédant déjà des checkpoints est refusée : utiliser un nouveau dossier.

## Voir le démasquage réel

```sh
JAX_PLATFORMS=cpu python3 demo.py --steps 16 --seed 42
JAX_PLATFORMS=cpu python3 demo.py --steps 8 --strategy confidence --text 'The ████ is ███████.'
```

Chaque ligne d'étape vient du modèle entraîné ; les caractères `█` indiquent les masques. Le texte généré par ces très petits modèles peut rester incorrect ou peu cohérent.

## Ce qui est implémenté

- Transformer pre-norm, attention causale ou bidirectionnelle, positions sinusoïdales, option de biais de distance fixe identique entre familles.
- Objectif **MDLM en temps discret fini** : poids exact `T * (m[k] - m[k-1]) / m[k]`, moyenne sur tous les caractères valides. La reconstruction finale est comprise dans le terme `k=1`.
- Sampler ancestral SUBS, conservation des caractères visibles, sorties sans tokens spéciaux.
- Samplers heuristiques à quotas : ordre aléatoire ou confiance du token proposé ; températures et top-k.
- Baseline MLM, masquage fixe 15 %, perte moyenne sur positions masquées. Ablation `unweighted` séparée.
- AR avec un vrai cache KV, validé numériquement contre le forward causal complet.
- AdamW, clipping, warmup, décroissance cosinus, PRNG explicite et reprise déterministe.
- Validation/test séparés ; NLL AR et estimation de NELBO finie correctement distinguées.
- Trois graines pour AR, MDLM et MLM ; calendriers linéaire/cosinus/logarithmique et perte non pondérée sur la graine 0.
- Latence synchronisée après chauffe, compilation séparée, processus neuf par cas et pic mémoire RSS ; longueur/batch variables.
- Métriques caractère : reconstruction, infilling, distinct-n, JS des n-grammes, répétition et copies exactes.

Le benchmark de génération compare **le même checkpoint avec plusieurs samplers**. Il ne réalise pas de distillation et ne prétend pas implémenter un modèle de flow matching.

## Structure

```text
nanodiff/model.py         Transformer et cache KV
nanodiff/diffusion.py     Processus, loss, samplers
nanodiff/training.py      AdamW et checkpoints NPZ
nanodiff/data.py          Splits, vocabulaire, empreinte du corpus
nanodiff/metrics.py       Baselines et mesures textuelles
nanodiff/evaluation.py    Évaluation et ablations
nanodiff/bench_worker.py  Chronométrage isolé et mémoire
nanodiff/reporting.py     Rapport et figures à partir des résultats
tests/                   Tests mathématiques, logiciels et intégration
configs/                 Protocole principal, smoke et pilotes
results/study/           Résultats principaux et checkpoints
results/quick/           Exploration initiale conservée
```

## Précautions d'interprétation

Une NELBO est une borne supérieure sur la NLL **en espérance** ; son estimation Monte Carlo n'est pas une NLL exacte, ni une borne certifiée pour chaque tirage. La NELBO ancestrale n'évalue pas la distribution de sortie du sampler heuristique par confiance. Les graphes le signalent explicitement.

Les réseaux reçoivent le même nombre de caractères, mais le nombre de cibles supervisées diffère. L'AR utilise le contexte gauche ; le MDLM peut utiliser les deux côtés. L'infilling n'est pas une comparaison AR/MDLM à information identique.

Les mesures à longueur 256 extrapolent un entraînement sur 128. La JS des trigrammes et distinct-n ne mesurent pas le sens. Trois graines sur un petit corpus ne justifient aucune conclusion à l'échelle LLM. Le pic RSS inclut le runtime et la compilation ; il ne représente pas la VRAM.

Le backend CPU a été testé sur un Apple M1 avec 8 Go. L'extension JAX Metal présente sur la machine échoue à l'initialisation ; les scripts choisissent donc CPU par défaut. Une installation JAX CUDA peut être utilisée via `JAX_PLATFORMS=cuda`, mais ce backend n'a pas été validé ici.

## Sources

- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [Code des auteurs de MDLM](https://github.com/kuleshov-group/mdlm)
- [Tiny Shakespeare, char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
- [Bonnes pratiques de benchmark JAX](https://docs.jax.dev/en/latest/benchmarking.html)

Le code de ce projet est une implémentation indépendante. Le corpus provient du dépôt char-rnn ; son URL et son SHA-256 sont dans chaque `manifest.json`.
