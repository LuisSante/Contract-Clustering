# Contract adaptation proposal

This is an integration design. The LegalXplain backend and frontend have not been modified.

## Preserve the reading workflow

The supplied tutorial shows authentication, demo/upload selection, ranked legal themes, a central PDF, relevant paragraph highlighting, paragraph explanations, bidirectional PDF/explanation selection, law-reference hover/click, deeper analysis using retrieved decisions, arbitrary paragraph selection, and feedback with attachments.

The existing [public](https://api.sta.legalxplain.visualdslab.com/docs) and [internal](https://api.sta.legalxplain.visualdslab.com/internal/docs) APIs complement that view: PDF analysis, RAG by theme and paragraph/page or geometry, samples, law detection, authentication, users, API keys and feedback. Publicly readable documentation does not make protected analysis endpoints unauthenticated.

## Replace the classification assumptions

The current system targets extraordinary appeals and a fixed set of STF themes. Contract adaptation needs a separate namespace and versioned registry. Existing numeric STF identifiers must not silently become contract class identifiers. Use a new API version or an explicit compatibility adapter.

Keep distinct fields for:

- **Product/instrument:** what document and financial product the user is reading; allow compound/unknown.
- **Clause themes:** one or more of the 14 provisional themes, each with actual evidence spans.
- **Paragraph roles:** definition, declaration, obligation, right, authorization, prohibition, condition, procedure or exception. Existing appeal labels Fact/Request/Argument should not be reused blindly.
- **Legal questions:** a question-specific assessment with supported/contradicted/not addressed/needs external evidence states; a quote is not proof of actual conduct.
- **Case context:** original DataJud subjects, source/version and separate allegation status. A contract alone does not contain every lawsuit allegation.
- **Quality/provenance:** full or partial/reconstructed source, redactions, model and taxonomy versions, processed-page coordinates and source mapping status.

Display the document's relevant clause themes on the left, keep clause evidence in the center, and explain the selected provision on the right. A contract-level theme list aggregates paragraph evidence, rather than forcing one class onto the whole PDF. Relevance scores are not legal certainty or calibrated probabilities unless separately validated.

## Retrieval and geometry

Use authoritative Brazilian consumer, banking, social-security and privacy sources as appropriate: CDC, applicable CMN/BCB rules, INSS rules when the payer/product falls within scope, LGPD and relevant STJ/TJBA decisions. Version legal sources by effective date and distinguish pending questions from settled holdings. The year in a CNJ case identifier is not the date of the contract. If the latter or the payer is unknown, expose that limitation rather than selecting a current rule automatically.

Existing STF decision retrieval is not a ready-made index for these questions. Store authority, jurisdiction, effective interval, official URL and status with each source. Explain what the contract says separately from what the source requires and which additional facts would be needed.

Processed PDFs can be reflowed/redacted. Text offsets from this pipeline refer to their extracted page text. Before adopting the legacy `page`, `x0/y0/x1/y1`, `width/height` contract, implement and verify word/span geometry on the exact PDF being displayed. Mapping to the original judicial PDF needs an explicit alignment; filename `pXX-YY` ranges do not establish a 1:1 mapping after reconstruction.

## Synthetic response sketch

```json
{
  "analysis_domain": "contracts.br.consumer_credit",
  "taxonomy_version": "0.1.0-proposal",
  "document_product": ["CARTAO_CREDITO_CONSIGNADO"],
  "source_completeness": "unknown_reconstructed",
  "contract_date": null,
  "case_subjects": [],
  "paragraphs": [{
    "paragraph_id": "synthetic-p1",
    "processed_page": 1,
    "themes": ["MARGEM_CONSIGNACAO", "FATURA_AMORTIZACAO"],
    "roles": ["AUTORIZACAO"],
    "text": "Autorizo o desconto mensal do pagamento mínimo da fatura.",
    "issue": "PAGAMENTO_MINIMO_AMORTIZA_TOTALMENTE",
    "evidence_status": "REQUER_CONTEXTO_EXTERNO",
    "explanation": "O trecho prevê desconto do mínimo; a evolução do saldo exige faturas, taxas e pagamentos.",
    "source_references": [],
    "geometry_status": "requires_mapping_to_displayed_pdf"
  }]
}
```

All text and IDs in this example are synthetic. No deployed API support for this new schema is claimed.
