"""Authorization boundaries, SecurityContext model, and tenant document access control."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from app.core.exceptions import ForbiddenError
from app.core.logging import get_logger

logger = get_logger("app.security.authorization")


class DocumentAction(StrEnum):
    """Permitted operations on document resources."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    PROCESS = "process"


@dataclass(frozen=True)
class SecurityContext:
    """Represents caller identity, tenant tenancy, and assigned authorization roles."""

    user_id: str
    tenant_id: str = "default"
    roles: tuple[str, ...] = field(default_factory=lambda: ("user",))
    is_authenticated: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.roles, list):
            object.__setattr__(self, "roles", tuple(self.roles))

    def is_admin(self) -> bool:
        """Check if caller has administrative privileges."""
        return "admin" in self.roles or "system" in self.roles

    def can_access_tenant(self, target_tenant: str) -> bool:
        """Check if caller is authorized to access resources belonging to target_tenant."""
        if self.is_admin():
            return True
        return self.tenant_id == target_tenant


@runtime_checkable
class AuthorizationPolicyProtocol(Protocol):
    """Protocol for resource-level authorization checks."""

    def authorize_document(
        self,
        context: SecurityContext,
        document_metadata: dict[str, Any],
        action: DocumentAction,
    ) -> bool:
        """Verify caller is authorized to perform action on the document.

        Args:
            context: Security context of caller.
            document_metadata: Metadata dictionary of target document.
            action: Requested DocumentAction.

        Returns:
            True if authorized.

        Raises:
            ForbiddenError: If unauthorized.
        """
        ...


class TenantDocumentAccessControl(AuthorizationPolicyProtocol):
    """Enforces strict multi-tenant isolation on documents and vector search candidates."""

    def __init__(self, tenant_enforcement_enabled: bool = True) -> None:
        self.tenant_enforcement_enabled = tenant_enforcement_enabled

    def authorize_document(
        self,
        context: SecurityContext,
        document_metadata: dict[str, Any],
        action: DocumentAction,
    ) -> bool:
        """Verify that caller tenant matches document tenant unless granted admin override."""
        if not self.tenant_enforcement_enabled:
            return True

        doc_tenant = document_metadata.get("tenant_id", "default")

        if not context.can_access_tenant(doc_tenant):
            logger.warning(
                f"Authorization denied for action '{action.value}': document belongs to tenant "
                f"'{doc_tenant}', but caller '{context.user_id}' is restricted to tenant '{context.tenant_id}'."
            )
            raise ForbiddenError(
                f"Access denied: Document belongs to tenant '{doc_tenant}', but caller is restricted to tenant '{context.tenant_id}'."
            )

        return True
