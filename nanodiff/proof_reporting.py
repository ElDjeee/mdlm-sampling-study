"""Evidence report for the analytically evaluable binary-copy benchmark."""
import json
from pathlib import Path
import numpy as np
from .reporting import Document
import matplotlib.pyplot as plt


def build_proof_report(out):
    out=Path(out)
    config=json.loads((out/'protocol.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    baselines=json.loads((out/'baselines.json').read_text())
    runs=summary['runs']
    histories=[json.loads((out/f'seed_{r["seed"]}'/'training.json').read_text())['history'] for r in runs]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for run,history in zip(runs,histories):
        steps=[row['step'] for row in history]
        axes[0].plot(steps,[100*r['valid_probability'] for r in history],marker='o',label=f'Graine {run["seed"]}')
        axes[1].plot(steps,[r['total_variation'] for r in history],marker='o',label=f'Graine {run["seed"]}')
    oracle=baselines['oracle'][str(config['diffusion_steps'])]
    axes[0].axhline(100*baselines['independent']['valid_probability'],ls=':',color='gray',label='Bits indépendants')
    axes[0].axhline(100*oracle['valid_probability'],ls='--',color='black',label='Débruiteur oracle, même sampler')
    axes[1].axhline(oracle['total_variation'],ls='--',color='black',label='Oracle, même sampler')
    axes[0].set(ylabel='Probabilité exacte de copie correcte (%)',xlabel='Mises à jour',ylim=(0,103))
    axes[1].set(ylabel='Distance exacte à la distribution cible (TV)',xlabel='Mises à jour',ylim=(0,1))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'proof_learning.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    labels=None
    for run in runs:
        with np.load(out/f'seed_{run["seed"]}'/'distributions.npz') as arrays:
            keep=arrays['target']>0
            labels=[''.join(map(str,row[:config['pairs']])) for row in arrays['states'][keep]]
            ax.plot(np.arange(len(labels)),arrays['model'][keep]*100,marker='o',label=f'Graine {run["seed"]}')
    ax.axhline(100/2**config['pairs'],color='black',ls='--',label='Cible uniforme')
    ax.set_xticks(np.arange(len(labels)),labels,rotation=45)
    ax.set(ylabel='Probabilité du mode (%)',xlabel='Première moitié ; la seconde doit être identique',ylim=(0,10))
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'proof_modes.png',dpi=160);plt.close(fig)
    doc=Document()
    doc.title('Preuve de fonctionnement : apprendre une distribution connue',1)
    doc.p('Verdict des critères fixés avant entraînement : '+('RÉUSSITE sur toutes les graines.' if summary['all_passed'] else 'ÉCHEC d’au moins un critère.'))
    doc.p('Cette expérience vérifie une propriété précise : un MDLM entraîné depuis zéro apprend-il les dépendances d’une distribution simple et son sampler les reproduit-il ? Les probabilités sont calculées exactement par énumération, pas déduites de quelques beaux exemples.')
    doc.title('Tâche et contrôle')
    doc.p('On tire quatre bits indépendants et équitables, puis on les répète : 0101 → 01010101. Les 16 séquences valides doivent toutes avoir une probabilité de 1/16. Des bits indépendants ont les mêmes marginales, mais ne respectent la copie que dans 1/16 = 6,25 % des générations. Générer toujours 00000000 respecte la règle tout en échouant au test de distribution et de diversité.')
    doc.p(f'Le Transformer existant est réutilisé sans modifier son code de réseau, sa loss MDLM ou son sampler ancestral. Il possède {runs[0]["parameters"]:,} paramètres, reçoit des masques absorbants et est entraîné pendant {config["train_steps"]} mises à jour, avec un batch de {config["batch_size"]}, trois graines et le CPU JAX. La largeur, le vocabulaire et la longueur sont adaptés à la tâche. Aucun oracle ne fournit de prédictions au réseau pendant l’entraînement ou la génération.')
    doc.title('Résultats exacts à 64 étapes')
    rows=[['Bits indépendants','6,25 %','0,9375','1,0000','—'],['Distribution cible','100 %','0','0,5000','—']]
    rows.append(['Débruiteur oracle',f'{100*oracle["valid_probability"]:.2f} %',f'{oracle["total_variation"]:.4f}',f'{oracle["exact_nll_bits_per_character"]:.4f}','—'])
    for run in runs:
        m=run['metrics']
        rows.append([f'MDLM, graine {run["seed"]}',f'{100*m["valid_probability"]:.2f} %',f'{m["total_variation"]:.4f}',f'{m["exact_nll_bits_per_character"]:.4f}',f'{run["train_seconds"]:.1f} s'])
    doc.table(['Modèle','Copie correcte','Distance TV ↓','NLL exacte, bits/car. ↓','Pas entraînement seuls'],rows)
    doc.p('TV = ½ Σx |pmodèle(x) − pcible(x)|, en sommant les 256 sorties binaires, y compris les séquences invalides. Les chiffres NLL sont ici de vraies vraisemblances, contrairement aux bornes NELBO du précédent benchmark textuel. L’entropie de la cible vaut 4 bits par séquence de 8 caractères, soit 0,5 bit/caractère.')
    doc.image('proof_learning.png','Les trois entraînements, évalués sans Monte Carlo : probabilité de validité et distance à la cible.')
    doc.title('Diversité et vérification du sampler réel')
    doc.image('proof_modes.png','Les probabilités des 16 modes valides : cette mesure empêche une copie toujours identique de passer pour un succès.')
    rows=[]
    for run in runs:
        check=run['sampler_check']
        rows.append([run['seed'],f'{100*check["empirical_valid_fraction"]:.2f} %',check['observed_valid_modes'],f'{check["empirical_vs_exact_tv"]:.4f}',f'{100*run["conditionals"]["determined_mask_accuracy"]:.2f} %'])
    doc.table(['Graine','Valides / 4096 tirages','Modes observés / 16','TV tirages / chaîne exacte','Reconstruction déterminable'],rows)
    doc.p('Les tirages proviennent de la fonction diffusion_sample déjà utilisée dans le projet, sans guidage ni correction de la copie. Leur distribution empirique est comparée au calcul exact indépendant. La reconstruction déterminable couvre exhaustivement les états bruités cohérents dont le bit partenaire est visible ; la réponse attendue est alors connue.')
    doc.code('Premiers tirages de la graine 0, sans sélection :\n'+'\n'.join(runs[0]['sampler_check']['first_16_samples']))
    doc.title('Pourquoi l’oracle lui-même reste sous 100 %')
    doc.p('En génération parallèle, les deux bits d’une paire peuvent être révélés à la même étape. S’ils étaient tous deux masqués, même un débruiteur parfait prédit deux marginales équitables ; elles sont tirées indépendamment et peuvent diverger. Pour S étapes linéaires, la probabilité exacte de validité de l’oracle est (1 − 1/(2S))⁴. Ce résidu vient du sampler fini ; il diminue lorsque S augmente.')
    rows=[]
    for steps in config['sampling_steps']:
        rows.append([steps,f'{100*baselines["oracle"][str(steps)]["valid_probability"]:.2f} %']+[f'{100*r["step_curve"][str(steps)]["valid_probability"]:.2f} %' for r in runs])
    doc.table(['Étapes','Oracle']+[f'Graine {r["seed"]}' for r in runs],rows)
    doc.title('Critères fixés et portée de la preuve')
    doc.code(json.dumps(config['acceptance'],indent=2))
    doc.p('Chaque graine doit passer tous les seuils au dernier checkpoint prévu. Les critères combinent validité, fidélité de distribution, vraisemblance exacte, probabilité minimale de chaque mode et accord sampler/énumération. Le protocole et les empreintes des sources sont conservés avec les résultats.')
    doc.p('C’est une preuve expérimentale de fonctionnement sur une tâche contrôlée. Les 16 modes sont présents dans la distribution d’entraînement : cela ne démontre pas une généralisation à de nouveaux modes, une compréhension du langage ou une reproduction du papier MDLM. Le précédent benchmark Shakespeare conserve ses limites. Le calcul dit exact énumère toute la chaîne en arithmétique flottante ; la conservation de masse est vérifiée à 10⁻⁸ près.')
    doc.title('Reproduire et auditer')
    doc.code('JAX_PLATFORMS=cpu python3 proof.py run\nJAX_PLATFORMS=cpu python3 check.py --output results/proof/tests.json\n# Pour repartir de zéro sans écraser les résultats :\nJAX_PLATFORMS=cpu python3 proof.py run --output results/proof_reproduction')
    doc.link('Résultats complets et critères','summary.json')
    doc.link('Protocole fixé avant entraînement','protocol.json')
    doc.link('Empreintes des sources','source_manifest.json')
    if (out/'tests.json').exists():
        tests=json.loads((out/'tests.json').read_text())
        doc.p(f'Tests logiciels et mathématiques exécutés : {tests["tests_run"]}. Réussite globale : {tests["successful"]}. Les tests spécifiques confrontent notamment l’énumération à la formule analytique de l’oracle et détectent l’effondrement sur un seul mode.')
        doc.link('Preuve d’exécution des tests','tests.json')
    doc.link('Code de l’énumération exacte','../../nanodiff/exact_copy.py')
    css='body{font:17px/1.6 system-ui;color:#203633;background:#f5f3eb;margin:0}main{max-width:1000px;padding:40px;margin:auto;background:#fffef9}h1{font-size:36px;line-height:1.2;color:#136b70}h2{margin-top:35px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #ccd7ce;padding:9px;text-align:left}th{background:#e7efe8}img{max-width:100%}figure{margin:25px 0}figcaption{font-size:13px}pre{background:#213b39;color:#f1f5e9;padding:18px;white-space:pre-wrap}a{color:#087b83}.table-wrap{overflow-x:auto}@media(max-width:650px){main{padding:20px}h1{font-size:28px}}'
    (out/'PROOF.md').write_text('\n'.join(doc.md))
    (out/'PROOF.html').write_text('<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Preuve de fonctionnement MDLM</title><style>'+css+'</style></head><body><main>'+''.join(doc.html)+'</main></body></html>')
    print(f'PROOF REPORT: {out/"PROOF.html"}',flush=True)
