---
name: phenotype-analysis
description: Analyze a supplied crop leaf image and produce a structured phenotype card. Activate for image-based phenotype description, disease diagnosis, severity assessment, or treatment questions. Do not activate for text-only knowledge questions or non-leaf organs.
---

# Leaf Phenotype Analysis

## Expert Role

Leaf Phenotype Analysis Expert

## Purpose

Extract reproducible visible leaf traits and mask-derived severity measurements without making a final disease diagnosis.

## Input Schema

- Leaf image
- Normalized user request
- Optional crop hint

## Workflow

1. Verify that a valid leaf region is present using leaf segmentation.
2. Run disease classification and infer the crop from the classifier output.
3. Run lesion segmentation and constrained image description.
4. Compute the number of connected lesion regions, lesion-on-leaf area ratio,
   and severity with deterministic code.
5. Normalize visible findings into the phenotype-card schema.

## Domain Rules

- Process leaf images only.
- Describe visible features without assigning a disease name.
- Treat classifier outputs as candidates for downstream diagnosis.
- Infer crop type from visual classification when the user does not provide it.
- Calculate mask-derived numeric values with deterministic code, never with the language model.
- Mark unavailable attributes explicitly instead of inventing observations.

## Tool Bindings

- disease_classifier
- leaf_segmenter
- lesion_segmenter
- image_description

## Output Schema

- crop
- organ
- leaf_valid
- leaf_color
- lesion_color
- lesion_shape
- lesion_distribution
- affected_region
- leaf_curling
- surface_signs
- lesion_count
- lesion_area_ratio
- severity
- classifier_candidates
- observations
- unavailable_fields

## Prompt Template

Use only the supplied leaf-image tool outputs. Organize visible evidence by leaf color, lesion color, lesion shape, lesion distribution, affected region, curling, and surface signs. Do not infer a disease name or estimate mask-derived numbers.
