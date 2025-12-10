# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability
Basel II requires banks to quantify and manage credit risk through three pillars. This means our model must be:
- **Interpretable**: Regulators and stakeholders need to understand how decisions are made
- **Well-documented**: Complete documentation of methodology, assumptions, and limitations
- **Validated**: Regular back-testing against actual outcomes
- **Risk-sensitive**: Accurately measure probability of default to determine capital requirements

### 2. Proxy Variable Necessity and Risks
**Why needed:** Our dataset has no direct "default" labels, only transaction data. We must infer risk from behavior patterns (RFM - Recency, Frequency, Monetary).

**Business risks:**
- **False Positives**: Good customers labeled high-risk → lost business
- **False Negatives**: Risky customers approved → potential defaults
- **Regulatory Risk**: Proxy may not meet regulatory standards
- **Fairness Issues**: Behavioral data might unintentionally discriminate

**Mitigation:**
- Conservative risk thresholds
- Human review for borderline cases
- Regular model validation

### 3. Simple vs Complex Model Trade-offs
**Simple Model (Logistic Regression):**
- ✅ Easy to explain, regulatory-friendly
- ✅ Stable, less overfitting
- ❌ May miss complex patterns

**Complex Model (Gradient Boosting):**
- ✅ Better predictive power
- ✅ Handles non-linear relationships
- ❌ Black box, hard to explain
- ❌ Regulatory scrutiny

**Our Approach:** Start with interpretable models, compare performance, prioritize explainability in regulated context.
