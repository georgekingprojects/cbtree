"""The home page of the app."""

from cbtree import styles
from cbtree.templates import template

import reflex as rx


@template(route="/", title="About", image="/treehacks_logo.jpeg")
def index() -> rx.Component:
    """The home page.

    Returns:
        The UI for the home page.
    """
    with open("README.md", encoding="utf-8") as readme:
        content = readme.read()
    return rx.markdown(content, component_map=styles.markdown_style)
