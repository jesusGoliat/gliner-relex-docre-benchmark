"""Construye el documento predicho (formato Re-DocRED, sin labels) a partir de las entidades agrupadas por ent_cluster."""


def build_doc(doc, entities):
    """doc: documento Re-DocRED (usa title y sents). Un tipo por entidad; types2 aparte."""
    vs = [[{'name': m['name'], 'sent_id': m['sent_id'], 'pos': list(m['pos']), 'type': e['type']}
           for m in e['mentions']] for e in entities]
    return {'title': doc['title'], 'sents': doc['sents'], 'vertexSet': vs,
            'types2': [list(e['types2']) for e in entities]}
