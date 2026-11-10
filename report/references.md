# References (verified Nov 10 2026)

## Prompt Injection & Multi-turn Attacks

1. **Greshake et al. (2023)**
   Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz.
   "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection."
   arXiv:2302.12173 [cs.CR].
   *Introduces indirect prompt injection, where adversarial instructions planted in retrieved data hijack LLM-integrated applications; provides a security-oriented taxonomy demonstrated on Bing Chat and GPT-4.*

2. **InjecAgent (Zhan et al., 2024)**
   Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang.
   "InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents."
   arXiv:2403.02691 [cs.CL]. ACL 2024 Findings.
   *A benchmark of 1,054 test cases across 17 user tools and 62 attacker tools measuring how often tool-using LLM agents follow indirect prompt injections (GPT-4 vulnerable 24% of the time).*

3. **AgentDojo (Debenedetti et al., 2024)**
   Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr.
   "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents."
   arXiv:2406.13352 [cs.CR].
   *An extensible evaluation environment with 97 tasks and 629 security test cases for measuring prompt injection attacks and defenses on tool-using agents.*

4. **Crescendo (Russinovich et al., 2024)**
   Mark Russinovich, Ahmed Salem, Ronen Eldan.
   "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack."
   arXiv:2404.01833 [cs.CR]. Accepted at USENIX Security 2025.
   *A multi-turn jailbreak that gradually escalates through references to the model's own replies, with an automated tool (Crescendomation) outperforming other jailbreak methods on AdvBench.*

## Positioning Note

We do NOT claim the first multi-turn injection benchmark. Our contributions are:
- Best-response (EGTA) evaluation of defenses against multi-turn injection
- Refusal-direction vs. injection-contrast steering comparison
- Measurement on Cohere multilingual models
