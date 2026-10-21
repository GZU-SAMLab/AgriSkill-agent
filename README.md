# AgriSkill-Agent

AgriSkill-Agent is a skill-guided and knowledge-enhanced single-agent framework for leaf-disease diagnosis and management. A central language model coordinates four procedural contracts for phenotype analysis, disease diagnosis, disease management, and report generation. Specialist visual models, deterministic measurements, hybrid literature retrieval, and a pesticide-use knowledge graph provide external evidence and tools.

This repository contains the core method code accompanying the manuscript. It is intended to document the framework rather than provide a standalone deployment package. Experimental records, model checkpoints, literature indexes, source images, external databases, service configuration, credentials, and demonstration components are excluded.

## Contents

```text
agriskill_agent/
  agent.py                 Central orchestration workflow
  planning.py              Skill activation and execution planning
  modules.py               Input normalization
  skills.py                Procedural implementation of the four skills
  skill_definitions.py     Procedural-contract loading
  integrations.py         Visual, retrieval, and knowledge-graph interfaces
  models.py                Structured inputs, evidence, and outputs
  terminology.py           Crop and disease terminology mapping
  config.py                Method parameters
  llm.py                   Central language-model interface
  tools.py                 Tool abstraction and registry
  skills/                  Procedural contracts
```

The released source focuses on the method described in the manuscript. External services and assets must be supplied separately to reproduce a deployment.
