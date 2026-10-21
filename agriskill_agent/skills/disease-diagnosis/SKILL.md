---
name: disease-diagnosis
description: Diagnose a leaf disease by integrating phenotype observations, visual classifier candidates, and hybrid literature retrieval. Activate for image-based diagnosis and text-only questions about disease identity, symptoms, causes, or differential diagnosis.
---

# Evidence-Grounded Disease Diagnosis

## Expert Role

Plant Disease Diagnosis Expert

## Purpose

Generate a primary diagnosis and plausible alternatives from visual and external knowledge evidence while keeping observations separate from inference.

## Input Schema

- Normalized user request
- Optional phenotype card
- Optional classifier candidates
- Retrieved literature evidence

## Workflow

1. Initialize the classifier Top-3 when an image is available.
2. Ask the center-agent model to generate one diagnostic-literature query from
   the user request, phenotype card, crop, and all crop-compatible Top-3 candidates.
3. Execute hybrid retrieval once with the generated query, while using crop,
   candidate disease, and phenotype/abstract metadata only as retrieval boundaries.
4. Compare the Top-3 using the visible phenotype and retrieved literature.
5. Produce the diagnosis, alternatives, rationale, limitations, and evidence identifiers.

## Domain Rules

- Restrict the final image diagnosis to classifier Top-3 or an evidence-limited
  abstention; never introduce a disease outside the supplied candidates.
- After crop inference, remove candidates assigned to another crop before
  retrieval planning and differential diagnosis.
- The center-agent model determines the natural-language RAG queries after
  reading the phenotype card and Top-3 candidates.
- Treat Top-1 as the initial hypothesis. Missing one expected sign is not enough
  to reject it.
- Retrieve comparative evidence across the crop-compatible candidates instead
  of retrieving evidence for only one alternative. An alternative may replace
  Top-1 only when the retrieved set covers both Top-1 and that alternative;
  one-sided evidence is insufficient to overturn Top-1.
- The model-generated natural-language query is passed unchanged to semantic
  retrieval, keyword retrieval, and reranking. Metadata restrictions define
  corpus scope but do not rewrite the query.
- When the query-planning model fails or returns invalid output, use a documented
  deterministic fallback query instead of stopping the pipeline.
- Do not continue image diagnosis when no valid leaf is detected.
- Ground diagnostic explanations in visible findings or supplied knowledge.
- Keep the two evidence roles explicit: phenotype-card fields are observations
  from the supplied image, while retrieved symptom descriptions are literature
  context used only to compare candidates.
- Never rewrite, extend, or relabel the phenotype card with a symptom found only
  in retrieved literature. Prefix rationale statements as "Observed:" or
  "Literature context:" when both kinds of evidence are discussed.
- State missing information instead of fabricating context.
- Raw classifier probabilities may be reported as model outputs.
- Do not create an agent-level confidence score or apply a confidence threshold.

## Tool Bindings

- hybrid_retrieval
- agent_model

## Output Schema

- primary_disease
- differential_diagnoses
- rationale
- limitations
- evidence_ids
- status

## Prompt Template

For an image case, first generate one literature query from the supplied phenotype card and classifier Top-3. After retrieval, choose the primary diagnosis only from those candidates or abstain when none is adequately supported. Treat Top-1 as an initial hypothesis rather than a fixed conclusion. Keep image observations separate from literature-reported typical symptoms: literature may support candidate comparison but must never be presented as a feature observed in the image. The pesticide-use knowledge graph is reserved for the management stage because it contains no symptom or pathogen relations. Preserve evidence identifiers and make limitations explicit.
