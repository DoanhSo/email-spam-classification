"""Shared package for the email spam classification project."""

from .models.decision_tree import DecisionTreeClassifier

__all__ = ["DecisionTreeClassifier"]
__version__ = "0.1.0"
