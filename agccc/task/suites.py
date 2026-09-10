"""The two acceptance suites (V3 §7.1).

These are the measurement instrument for the whole experiment. They are fixed,
versioned, identical across every arm and every run, and are **never** generated
by an agent. Nothing here may import from `governance`, `strategies` or
`agents`: a suite that could see who was being measured would not be a suite.

Each check is a predicate over the artefact:

    passes  <=>  required endpoint implemented  AND  its defect absent

so a suite result is reconstructible by hand from an artefact, which is what
makes the derived confidence in `awareness/confidence.py` auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .artefact import Artefact
from .defects import Defect, Endpoint

SUITE_VERSION = "1.0.0"


@dataclass(frozen=True)
class Check:
    id: str
    description: str
    endpoint: Endpoint
    defect: Defect


FUNCTIONAL: tuple[Check, ...] = (
    Check("F1", "registration succeeds", Endpoint.REGISTER, Defect.REGISTER_BROKEN),
    Check("F2", "duplicate registration rejected", Endpoint.REGISTER, Defect.DUPLICATE_NOT_REJECTED),
    Check("F3", "login with correct credentials issues a token", Endpoint.LOGIN, Defect.LOGIN_BROKEN),
    Check("F4", "login with wrong password rejected", Endpoint.LOGIN, Defect.WRONG_PASSWORD_ACCEPTED),
    Check("F5", "/me with a valid token returns identity", Endpoint.ME, Defect.ME_BROKEN),
    Check("F6", "/me without a token returns 401", Endpoint.ME, Defect.ME_MISSING_401),
    Check("F7", "refresh issues a new token", Endpoint.REFRESH, Defect.REFRESH_BROKEN),
    Check("F8", "logout invalidates the token", Endpoint.LOGOUT, Defect.LOGOUT_NOT_INVALIDATING),
)

SECURITY: tuple[Check, ...] = (
    Check("S1", "weak passwords rejected", Endpoint.REGISTER, Defect.WEAK_PASSWORD_ACCEPTED),
    Check("S2", "login path resists SQL/NoSQL injection", Endpoint.LOGIN, Defect.SQLI_LOGIN),
    Check("S3", "/me verifies token claims (no authz bypass)", Endpoint.ME, Defect.NO_AUTHZ_ON_ME),
    Check("S4", "JWT alg=none rejected", Endpoint.ME, Defect.ALG_NONE_ACCEPTED),
    Check("S5", "JWT signed with a wrong key rejected", Endpoint.ME, Defect.WRONG_KEY_ACCEPTED),
    Check("S6", "passwords stored irreversibly hashed", Endpoint.REGISTER, Defect.PLAINTEXT_PASSWORD),
    Check("S7", "no user enumeration via differential errors", Endpoint.LOGIN, Defect.USER_ENUMERATION),
)


@dataclass(frozen=True)
class SuiteResult:
    name: str
    passed: int
    total: int
    failures: tuple[str, ...]

    @property
    def fraction(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def green(self) -> bool:
        return self.passed == self.total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "total": self.total,
            "failures": list(self.failures),
        }


def _run(name: str, checks: tuple[Check, ...], artefact: Artefact) -> SuiteResult:
    failures = tuple(
        c.id
        for c in checks
        if c.endpoint not in artefact.endpoints or c.defect in artefact.defects
    )
    return SuiteResult(name, len(checks) - len(failures), len(checks), failures)


def functional_suite(artefact: Artefact) -> SuiteResult:
    return _run("functional", FUNCTIONAL, artefact)


def security_suite(artefact: Artefact) -> SuiteResult:
    return _run("security", SECURITY, artefact)
