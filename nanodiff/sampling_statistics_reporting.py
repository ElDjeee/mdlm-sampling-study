"""Insert measured sampling statistics alongside the existing inference results."""
import json
import numpy as np
import matplotlib.pyplot as plt


def append_sampling_statistics(doc, out):
    path = out/'sampling_statistics/summary.json'
    if not path.exists():
        return
    result = json.loads(path.read_text())
    protocol = result['protocol']
    steps = protocol['steps']
    curves = result['curves']['js3']
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ci, label in enumerate(protocol['checkpoints']):
        axes[0].plot(steps,[curves[str(s)]['checkpoint_means'][ci] for s in steps],
                     lw=1,alpha=.55,label=label)
    means = np.array([curves[str(s)]['mean'] for s in steps])
    lows = np.array([curves[str(s)]['ci95'][0] for s in steps])
    highs = np.array([curves[str(s)]['ci95'][1] for s in steps])
    axes[0].errorbar(steps,means,yerr=[means-lows,highs-means],color='#24312f',
                     marker='o',capsize=4,lw=2,label='Moyenne · IC 95 % conditionnel')
    axes[0].set(xscale='log',xlabel='Étapes ancestrales',ylabel='JS des trigrammes (bits)',
                title='Nouvelles générations · trois checkpoints')
    axes[0].set_xticks(steps,labels=list(map(str,steps)))
    axes[0].legend(fontsize=8)
    lags = protocol['pair_lags']
    contrasts = [result['contrasts'][f'pair_js_lag{lag}'] for lag in lags]
    means = np.array([r['mean'] for r in contrasts])
    lows = np.array([r['ci95'][0] for r in contrasts])
    highs = np.array([r['ci95'][1] for r in contrasts])
    axes[1].errorbar(lags,means,yerr=[means-lows,highs-means],color='#bc4f26',marker='o',capsize=4)
    axes[1].axhline(0,color='#555',ls='--',lw=1)
    axes[1].set(xscale='log',xlabel='Distance entre caractères',ylabel='Δ JS des paires : 64 − 4 étapes',
                title='Cooccurrences · négatif = rapprochement')
    axes[1].set_xticks(lags,labels=list(map(str,lags)))
    for ax in axes:
        ax.grid(alpha=.2)
    figure = 'figures/sampling_statistics.png'
    fig.savefig(out/figure,dpi=160)
    plt.close(fig)
    doc.html.append('<section id="sampling-statistics">')
    doc.title('6.1 Effet du nombre d’étapes : générations répétées',3)
    doc.p(f'Le premier balayage ne comportait que 32 textes par réglage. Pour vérifier si son écart entre '
          f'4 et 64 étapes dépasse la variabilité du tirage, nous avons généré {result["total_sequences"]:,} '
          f'nouveaux textes : {len(protocol["checkpoints"])} checkpoints existants × {len(steps)} nombres '
          f'd’étapes × {protocol["replicates"]} répétitions × {protocol["samples_per_replicate"]} séquences '
          f'de {protocol["length"]} caractères. Température 1, aucun top-k, sampler ancestral inchangé. '
          'Les répétitions sont appariées par clé aléatoire entre nombres d’étapes ; les clés diffèrent '
          'entre répétitions et checkpoints.')
    doc.p(f'La référence est fixe : {result["reference_sequences"]} fenêtres disjointes de validation '
          f'({result["reference_chars"]:,} caractères). Le critère principal est la variation de JS des '
          'trigrammes entre 64 et 4 étapes. Ce suivi reste exploratoire : cette comparaison a été choisie '
          'après le premier balayage, puis son protocole a été figé avant les nouvelles générations. '
          'Les valeurs de JS empiriques dépendent du nombre de textes et ne se comparent pas directement '
          'aux valeurs du balayage à 32 textes.')
    doc.image(figure,'Gauche : JS3 moyenne sur les répétitions, avec moyennes par checkpoint. '
              'Droite : variation de JS des cooccurrences à différentes distances. IC bootstrap '
              'ponctuels à 95 %, conditionnels aux trois checkpoints et au corpus de référence.')
    rows=[]
    for s in steps:
        value=curves[str(s)]
        rows.append([s,f'{value["mean"]:.4f}',
                     f'[{value["ci95"][0]:.4f} ; {value["ci95"][1]:.4f}]',
                     f'{result["curves"]["js1"][str(s)]["mean"]:.4f}'])
    doc.table(['Étapes','JS3 moyenne','IC 95 % conditionnel','JS unigrammes moyenne'],rows)
    primary=result['primary']
    sign=primary['sign_test']
    low,high=primary['ci95']
    doc.p(f'Le contraste moyen JS3(64) − JS3(4) vaut {primary["mean"]:+.4f} bit '
          f'(IC 95 % [{low:+.4f} ; {high:+.4f}]). '
          f'{sign["negative"]} paires donnent une diminution, {sign["positive"]} une augmentation '
          f'et {sign["ties"]} une égalité. Le test exact bilatéral des signes donne '
          f'p = {sign["p_two_sided"]:.3g}. '
          + ('L’intervalle entièrement négatif indique un rapprochement des statistiques locales '
             'avec davantage d’étapes, dans ce dispositif. ' if high<0 else
             'L’intervalle ne permet pas de conclure à un rapprochement systématique dans ce dispositif. ')
          + 'Cela ne démontre ni une bonne qualité linguistique, ni une reproduction de la perplexité du papier.')
    doc.p('Les intervalles utilisent 10 000 rééchantillonnages de lots entiers, séparément dans chaque '
          'checkpoint, et non des caractères supposés indépendants. Le test des signes suppose, sous '
          'l’hypothèse nulle, des signes équiprobables et indépendants entre paires. Ni le bootstrap ni '
          'ce test ne mesurent l’incertitude sur de nouveaux entraînements : les trois checkpoints '
          'sont conditionnés, pas traités comme 24 modèles indépendants. Les diagnostics de paires '
          'sont secondaires, avec intervalles ponctuels sans correction pour comparaisons multiples.')
    unigram=result['contrasts']['js1']
    near=result['contrasts']['pair_js_lag1']
    far=result['contrasts']['pair_js_lag16']
    doc.p(f'Le profil spatial nuance ce gain : la JS des paires voisines diminue de {near["mean"]:.4f}, '
          f'alors qu’à distance 16 elle augmente de {far["mean"]:+.4f}. Le contraste sur les unigrammes '
          f'est proche de zéro ({unigram["mean"]:+.5f}, IC 95 % '
          f'[{unigram["ci95"][0]:+.5f} ; {unigram["ci95"][1]:+.5f}]). '
          'Ces diagnostics exploratoires suggèrent un bénéfice concentré sur les arrangements locaux, '
          'pas une amélioration uniforme à toutes les distances. Ils ne permettent pas d’affirmer '
          'l’égalité des marginales ni l’apprentissage de dépendances longues ; le biais architectural '
          'de localité et le faible budget d’entraînement restent des limites de cette interprétation.')
    doc.p('À chaque transition ancestrale, les nouveaux caractères sont tirés indépendamment '
          'conditionnellement au texte visible, puis ne sont plus modifiés. Changer le nombre d’étapes '
          'modifie donc le contexte disponible lors de ces décisions. La mesure des cooccurrences '
          'examine l’effet statistique de ce changement ; elle reste sensible aux fréquences marginales '
          'et n’isole pas, à elle seule, une perte de dépendances. Les checkpoints étant identiques '
          'entre réglages, le contraste mesure un effet du sampling à entraînement fixé, pas '
          'un effet d’apprentissage ni un mécanisme causal identifié.')
    doc.link('Protocole figé, sources et checkpoints identifiés','sampling_statistics/protocol.json')
    doc.link('Mesures des 120 lots de génération','sampling_statistics/batch_metrics.json')
    doc.link('Contrastes, intervalles et empreintes des générations','sampling_statistics/summary.json')
    doc.html.append('</section>')
