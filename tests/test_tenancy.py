"""Tests for TenantContext (REL-02)."""

import pytest
from pydantic import ValidationError

from schemas.tenancy import TenantContext


def test_unscoped_context_authorizes_any_project_and_application() -> None:
    context = TenantContext(tenant_id="t-acme")
    assert context.authorizes(project_id="p-1", application_id="a-1") is True
    assert context.authorizes(project_id="p-2", application_id="a-2") is True


def test_project_scoped_context_rejects_a_different_project() -> None:
    context = TenantContext(tenant_id="t-acme", project_id="p-1")
    assert context.authorizes(project_id="p-1", application_id="a-1") is True
    assert context.authorizes(project_id="p-2", application_id="a-1") is False


def test_application_scoped_context_rejects_a_different_application() -> None:
    context = TenantContext(tenant_id="t-acme", application_id="a-1")
    assert context.authorizes(project_id="p-1", application_id="a-1") is True
    assert context.authorizes(project_id="p-1", application_id="a-2") is False


def test_fully_scoped_context_requires_exact_match_on_both() -> None:
    context = TenantContext(tenant_id="t-acme", project_id="p-1", application_id="a-1")
    assert context.authorizes(project_id="p-1", application_id="a-1") is True
    assert context.authorizes(project_id="p-1", application_id="a-2") is False
    assert context.authorizes(project_id="p-2", application_id="a-1") is False


def test_empty_tenant_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TenantContext(tenant_id="")
