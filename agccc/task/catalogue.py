"""The authored task specifications.

Three tasks, authored to be *structurally* different rather than cosmetically
different. A second task that merely renamed the first one's units would test
nothing: the interesting question is whether the ordering between arms survives
a task whose shape makes the specialist less decisive, and whether it survives
one where the security surface is small but sharp.

The axes varied across the three, and why each matters:

  units            more units means a planner omission costs more and REPLAN is
                   worth more, which changes what the scorer should prefer
  suite ratio      the derived confidence is 0.5*F + 0.5*S, so an 8/7 split and
                   a 6/9 split weight one repaired defect very differently
  asymmetry        `p_fix_security_coder` vs `p_fix_security_specialist` is the
                   gap that makes getting the specialist into the topology
                   decisive. REST auth has a wide gap (0.15 vs 0.90); ETL has a
                   deliberately narrow one (0.40 vs 0.75), so an arm that cannot
                   restructure is punished far less there.

REST_AUTH reproduces the original task exactly — same unit strings, same defect
strings, same order. The seeded draws are keyed on those strings, so its results
after this refactor must be identical to the results reported before it. That is
checked, not assumed.
"""

from __future__ import annotations

from .spec import Check, TaskSpec

# ---------------------------------------------------------------------------
# 1. REST authentication API  (the original task; 5 units, 8F/7S, wide gap)
# ---------------------------------------------------------------------------

REST_AUTH = TaskSpec(
    key="rest_auth",
    name="REST authentication API",
    units=("POST /register", "POST /login", "GET /me", "POST /refresh", "POST /logout"),
    functional=(
        Check("F1", "registration succeeds", "POST /register", "register_broken"),
        Check("F2", "duplicate registration rejected", "POST /register", "duplicate_not_rejected"),
        Check("F3", "login with correct credentials issues a token", "POST /login", "login_broken"),
        Check("F4", "login with wrong password rejected", "POST /login", "wrong_password_accepted"),
        Check("F5", "/me with a valid token returns identity", "GET /me", "me_broken"),
        Check("F6", "/me without a token returns 401", "GET /me", "me_missing_401"),
        Check("F7", "refresh issues a new token", "POST /refresh", "refresh_broken"),
        Check("F8", "logout invalidates the token", "POST /logout", "logout_not_invalidating"),
    ),
    security=(
        Check("S1", "weak passwords rejected", "POST /register", "weak_password_accepted"),
        Check("S2", "login path resists SQL/NoSQL injection", "POST /login", "sqli_login"),
        Check("S3", "/me verifies token claims (no authz bypass)", "GET /me", "no_authz_on_me"),
        Check("S4", "JWT alg=none rejected", "GET /me", "alg_none_accepted"),
        Check("S5", "JWT signed with a wrong key rejected", "GET /me", "wrong_key_accepted"),
        Check("S6", "passwords stored irreversibly hashed", "POST /register", "plaintext_password"),
        Check("S7", "no user enumeration via differential errors", "POST /login", "user_enumeration"),
    ),
)

# ---------------------------------------------------------------------------
# 2. Batch ingestion pipeline  (8 units, 6F/9S, NARROW competence gap)
#
# The hard case for the architecture. Security-heavy suite, but a Coder that is
# competent at security work (0.40 against REST auth's 0.15) and a specialist
# that is merely good rather than excellent (0.75 against 0.90). If governed
# re-orchestration only wins because the specialist is irreplaceable, it should
# win by much less here.
# ---------------------------------------------------------------------------

ETL_INGEST = TaskSpec(
    key="etl_ingest",
    name="Batch data-ingestion pipeline",
    units=(
        "ingest.connect", "ingest.parse", "ingest.validate", "ingest.transform",
        "ingest.dedupe", "ingest.load", "ingest.audit", "ingest.replay",
    ),
    functional=(
        Check("F1", "source connection established", "ingest.connect", "source_connect_broken"),
        Check("F2", "malformed records rejected", "ingest.parse", "parse_accepts_malformed"),
        Check("F3", "schema validation enforced", "ingest.validate", "schema_validation_skipped"),
        Check("F4", "transform preserves row count", "ingest.transform", "transform_drops_rows"),
        Check("F5", "duplicate keys collapse correctly", "ingest.dedupe", "dedupe_false_merge"),
        Check("F6", "load is atomic across partitions", "ingest.load", "load_not_atomic"),
    ),
    security=(
        Check("S1", "PII columns masked before load", "ingest.load", "pii_not_masked"),
        Check("S2", "source identifiers resist injection", "ingest.connect", "injection_via_source_name"),
        Check("S3", "credentials absent from logs", "ingest.audit", "credentials_in_logs"),
        Check("S4", "unsigned sources rejected", "ingest.connect", "unsigned_source_accepted"),
        Check("S5", "file sources resist path traversal", "ingest.parse", "path_traversal_on_source"),
        Check("S6", "audit log is append-only", "ingest.audit", "audit_log_tamperable"),
        Check("S7", "replay is idempotent", "ingest.replay", "replay_not_idempotent"),
        Check("S8", "oversize payloads rejected", "ingest.parse", "oversize_payload_accepted"),
        Check("S9", "TLS verification enforced", "ingest.connect", "tls_verification_disabled"),
    ),
    world_overrides={
        "p_defect_functional": 0.35,
        "p_defect_security": 0.55,
        "p_fix_security_coder": 0.40,        # narrow gap: the Coder can do security work
        "p_fix_security_specialist": 0.75,   # and the specialist is merely good
        "p_planner_omits_endpoint": 0.14,    # more units, more to omit
    },
)

# ---------------------------------------------------------------------------
# 3. Checkout and payment flow  (4 units, 10F/4S, wide gap, dense functional load)
#
# The inverse shape: few units, a large functional suite and a small security
# suite where the Coder is nearly helpless (0.10). Functional defects dominate
# the early rounds and the security surface is what ends the run.
# ---------------------------------------------------------------------------

CHECKOUT_FLOW = TaskSpec(
    key="checkout_flow",
    name="Checkout and payment flow",
    units=("checkout.cart", "checkout.price", "checkout.pay", "checkout.receipt"),
    functional=(
        Check("F1", "cart total is correct", "checkout.cart", "cart_total_wrong"),
        Check("F2", "merging carts preserves items", "checkout.cart", "cart_merge_loses_items"),
        Check("F3", "stock reserved before capture", "checkout.cart", "stock_not_reserved"),
        Check("F4", "discounts applied once", "checkout.price", "discount_not_applied"),
        Check("F5", "tax computed for the destination", "checkout.price", "tax_missing"),
        Check("F6", "currency rounding is exact", "checkout.price", "currency_rounding_error"),
        Check("F7", "concurrent price updates serialised", "checkout.price", "price_race"),
        Check("F8", "partial capture supported", "checkout.pay", "partial_capture_broken"),
        Check("F9", "refund returns the full amount", "checkout.pay", "refund_broken"),
        Check("F10", "receipt lists every line item", "checkout.receipt", "receipt_missing_items"),
    ),
    security=(
        Check("S1", "replayed charges rejected", "checkout.pay", "replay_charge_accepted"),
        Check("S2", "idempotency key honoured", "checkout.pay", "idempotency_key_ignored"),
        Check("S3", "card data never logged", "checkout.receipt", "card_data_logged"),
        Check("S4", "webhook signatures verified", "checkout.pay", "webhook_signature_unverified"),
    ),
    world_overrides={
        "p_defect_functional": 0.45,
        "p_defect_security": 0.70,
        "p_fix_security_coder": 0.10,        # wide gap: payment security is specialist work
        "p_fix_security_specialist": 0.95,
        "p_planner_omits_endpoint": 0.06,    # few units, little to omit
    },
)

TASKS: dict[str, TaskSpec] = {t.key: t for t in (REST_AUTH, ETL_INGEST, CHECKOUT_FLOW)}
DEFAULT_TASK = REST_AUTH
