from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def get_revision_prompt() -> ChatPromptTemplate:
    system = (
        "You revise an existing Georgia travel itinerary.\n"
        "Apply only the changes explicitly requested in the user's feedback.\n"
        "Keep unchanged days and activities stable unless the feedback requires a change.\n"
        "Do not invent new constraints or extra changes.\n"
        "Preserve the day-by-day structure and the original trip length unless the feedback asks otherwise."
    )

    human = (
        "User feedback:\n{feedback}\n\n"
        "Current plan:\n{current_plan}\n\n"
        "Trip parameters:\n{trip_parameters}\n\n"
        "Revise the itinerary accordingly."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )


build_revision_prompt = get_revision_prompt
load_prompt = get_revision_prompt

