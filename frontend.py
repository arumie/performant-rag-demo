from collections.abc import Callable
from dataclasses import dataclass

import requests
from nicegui import run, ui

from app.types.db import SourceOutput
from app.types.draft import DraftOutput


@dataclass
class State:
    on_change: Callable
    version = "1"
    user_id = "user456"
    email_body = "Hi\nWhat are the main features of Fake Product?\nThanks"
    result: DraftOutput | None = None
    loading = False

    def set_preset(self, preset: int) -> None:
        presets = {
            "1": {
                1: "Hi\nWhat are the main features of Fake Product?\nThanks",
                2: "Hi\n\nHow secure is Fake Product?\nThanks",
                3: "Hi\n\nWhat are the pricing plans for Fake Product?\nThanks",
                4: "Hi\n\nCan Fake Product be customized?\nThanks",
            },
            "2": {
                1: "Hi\nWhat are the main features of Fake Product?\nThanks",
                2: "Hello\nWhat are the pricing plans for Fake Product 2.0?\nThanks",
                3: "Hi\nWhat are the differences in features for Fake Product and Fake Product 2.0?\nThanks",
                4: "Hi\nWhats the price for Fake Product and Fake Product 2.0?\nThanks",
            },
            "3": {
                1: "Hi\nWhat is the price for the Basic plan?\nThanks",
                2: "Hi\n\nHow does the Fake Product integrate with existing systems?\nThanks",
                3: "Hi\n\nWhat are the pricing plans for Fake Product?\nThanks",
                4: "Hi\n\nCan Fake Product be customized?\nThanks",
            },
            "4": {
                1: "Hi\nI'm very unhappy with the Fake Product. Can i get a refund?\nThanks",
                2: "Hi\nWhen we talked last time you helped me with setting up the Fake Product plan. Can you remind me how that worked?\nThanks",
                3: "Hi\nWhen does my current plan end?\nThanks",
                4: "Hi\nWhat are the price plans for Fake Product? Also what is my current plan, and the amount I pay for it?\nThanks",
            },
        }

        if self.version in presets:
            self.email_body = presets[self.version].get(preset, "")
        self.on_change()

    def clear_result(self) -> None:
        self.result = None
        self.on_change()

    async def create_draft(self) -> None:
        ui.notify("Creating draft...")
        self.loading = True
        self.result = None
        self.on_change()

        url = f"http://localhost:8000/v{state.version}/draft"
        print(url, state)
        response = await run.io_bound(
            requests.post, url, timeout=40, json={"from_user": state.user_id, "email_body": state.email_body},
        )
        self.loading = False
        result = DraftOutput(**response.json())
        self.result = result
        self.on_change()
        ui.notify("Draft created")


@ui.refreshable
def main() -> None:
    with ui.card().classes("w-2/3 m-auto"), ui.row().classes("w-full"):
        ui.button("Preset 1", on_click=lambda: state.set_preset(1))
        ui.button("Preset 2", on_click=lambda: state.set_preset(2))
        ui.button("Preset 3", on_click=lambda: state.set_preset(3))
        ui.button("Preset 4", on_click=lambda: state.set_preset(4))
        ui.space()
        ui.button(
            "Database",
            on_click=lambda: ui.navigate.to(
                target=f"http://localhost:6333/dashboard#/collections/V{'1' if state.version == '4' else state.version}",
                new_tab=True,
            ),
        )
    with ui.card().classes("w-2/3 m-auto"):
        with ui.row().classes("w-full"):
            ui.select(
                label="Version",
                options=["1", "2", "3", "4"],
                value=state.version,
                on_change=lambda: state.set_preset(1),
            ).bind_value(state, "version").classes("w-44")
            if state.version == "4":
                ui.select(
                    label="User",
                    options=["user123", "user456", "user789", "user999"],
                    value=state.user_id,
                ).bind_value(state, "user_id").classes("w-44")
        ui.textarea("Message", value=state.email_body).classes("w-full").bind_value(state, "email_body")
        ui.button("Create draft", on_click=lambda: state.create_draft(), color="green").classes("w-full")

    with ui.card().classes("w-2/3 m-auto"):
        if state.loading:
            ui.spinner(size="3em").classes("m-auto")
        elif state.result:
            ui.button("X", on_click=lambda: state.clear_result(), color="red").classes("rounded-full ml-auto")
            if state.result.fail_reason:
                ui.markdown("#### Draft not created")
                render_markdown(f"{state.result.fail_reason}").classes("text-red-500")
            else:
                ui.markdown("#### Draft")
                render_markdown(f"{state.result.draft}")
                render_sources(state.result.sources)
                render_questions()
        else:
            render_markdown("No result yet\nClick 'Create draft' to generate a draft")

    ui.run(title="Performant RAG demo")


def render_markdown(text: str) -> None:
    ui.markdown(text.replace("\n", "<br>").replace("_", r"\_"))


def render_questions() -> None:
    if state.result.questions and len(state.result.questions) > 0:
        with ui.expansion("Questions", icon="quiz").classes("w-full"):
            for question in state.result.questions:
                ui.separator()
                render_markdown(f"Question: {question.question}")
                render_markdown(f"Answer: {question.answer}")
                render_sources(question.sources)


def render_sources(sources: list[SourceOutput]) -> None:
    if sources and len(sources) > 0:
        with ui.expansion("Sources", icon="source").classes("w-full"):
            for source in sources:
                ui.separator()
                if source.text:
                    ui.markdown("##### Text")
                    render_markdown(f"{source.text}")
                if source.question:
                    ui.markdown("##### Question")
                    render_markdown(f"{source.question}")


state = State(on_change=main.refresh)
main()
