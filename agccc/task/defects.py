"""The defect catalogue for the REST authentication task (V3 §7).

Every defect maps 1:1 onto exactly one assertion in one acceptance suite, so a
suite result can be read back as a defect set and checked by hand. This is what
makes the confidence figure auditable rather than asserted.

Functional and security defects are kept disjoint. A missing authorisation check
is arguably both a functional contract violation and a security hole, and a
real test suite would catch it twice; here `ME_MISSING_401` (no token at all is
accepted) and `NO_AUTHZ_ON_ME` (a token is present but its claims are not
verified) are deliberately separated so that the two suites never share a cause.
Coupled suites would make the derived-confidence weights meaningless.
"""

from __future__ import annotations

from enum import Enum


class Endpoint(str, Enum):
    REGISTER = "POST /register"
    LOGIN = "POST /login"
    ME = "GET /me"
    REFRESH = "POST /refresh"
    LOGOUT = "POST /logout"


ALL_ENDPOINTS: frozenset[Endpoint] = frozenset(Endpoint)


class Defect(str, Enum):
    # --- functional (8) -------------------------------------------------
    REGISTER_BROKEN = "register_broken"
    DUPLICATE_NOT_REJECTED = "duplicate_not_rejected"
    LOGIN_BROKEN = "login_broken"
    WRONG_PASSWORD_ACCEPTED = "wrong_password_accepted"
    ME_BROKEN = "me_broken"
    ME_MISSING_401 = "me_missing_401"
    REFRESH_BROKEN = "refresh_broken"
    LOGOUT_NOT_INVALIDATING = "logout_not_invalidating"

    # --- security (7) ---------------------------------------------------
    WEAK_PASSWORD_ACCEPTED = "weak_password_accepted"
    SQLI_LOGIN = "sqli_login"
    NO_AUTHZ_ON_ME = "no_authz_on_me"
    ALG_NONE_ACCEPTED = "alg_none_accepted"
    WRONG_KEY_ACCEPTED = "wrong_key_accepted"
    PLAINTEXT_PASSWORD = "plaintext_password"
    USER_ENUMERATION = "user_enumeration"


FUNCTIONAL_DEFECTS: tuple[Defect, ...] = (
    Defect.REGISTER_BROKEN,
    Defect.DUPLICATE_NOT_REJECTED,
    Defect.LOGIN_BROKEN,
    Defect.WRONG_PASSWORD_ACCEPTED,
    Defect.ME_BROKEN,
    Defect.ME_MISSING_401,
    Defect.REFRESH_BROKEN,
    Defect.LOGOUT_NOT_INVALIDATING,
)

SECURITY_DEFECTS: tuple[Defect, ...] = (
    Defect.WEAK_PASSWORD_ACCEPTED,
    Defect.SQLI_LOGIN,
    Defect.NO_AUTHZ_ON_ME,
    Defect.ALG_NONE_ACCEPTED,
    Defect.WRONG_KEY_ACCEPTED,
    Defect.PLAINTEXT_PASSWORD,
    Defect.USER_ENUMERATION,
)

assert not set(FUNCTIONAL_DEFECTS) & set(SECURITY_DEFECTS), "suites must not share a cause"
assert set(FUNCTIONAL_DEFECTS) | set(SECURITY_DEFECTS) == set(Defect)


def is_security(defect: Defect) -> bool:
    return defect in SECURITY_DEFECTS
