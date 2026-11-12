import ast
import gradio as gr
from openai import OpenAI, APIConnectionError, APITimeoutError, APIError

# ============================================================
# НАЛАШТУВАННЯ ЛОКАЛЬНОЇ МОДЕЛІ OLLAMA
# ============================================================

MODEL = "qwen3:1.7b"
BASE_URL = "http://localhost:11434/v1"

TEMPERATURE = 0.4
MAX_TOKENS = 1024
CONTEXT_LENGTH = 4096

client = OpenAI(
    base_url=BASE_URL,
    api_key="ollama",
    timeout=120.0
)

SYSTEM_PROMPT = """
Ти AI Code Assistant — помічник студента-програміста.
Відповідай українською мовою.
Пояснюй складні речі простими словами.
Якщо надаєш код, використовуй Markdown форматування.
Не вигадуй результати виконання коду.
Якщо не впевнений у відповіді, повідом про це.
Під час перевірки коду розрізняй синтаксичні та логічні помилки.
Не вигадуй помилки, яких немає у наданому коді.
"""


# ============================================================
# РОБОТА З МОДЕЛЛЮ
# ============================================================

def ask_model(messages):
    """Надсилає повідомлення локальній моделі через Ollama API."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
        extra_body={
            "options": {
                "num_ctx": CONTEXT_LENGTH
            }
        }
    )

    if not response or not response.choices:
        raise ValueError("Сервер повернув відповідь у невідомому форматі.")

    answer = response.choices[0].message.content

    if not isinstance(answer, str) or not answer.strip():
        return "Модель повернула порожню відповідь. Спробуй сформулювати запит інакше."

    return answer.strip()


def handle_error(error):
    """Перетворює технічні помилки на зрозумілі повідомлення."""
    if isinstance(error, APIConnectionError):
        return (
            "### Не вдалося підключитися до Ollama\n\n"
            "Перевір, чи запущений Ollama. Також переконайся, "
            "що локальний API доступний за адресою "
            "`http://localhost:11434`."
        )

    if isinstance(error, APITimeoutError):
        return (
            "### Час очікування вичерпано\n\n"
            "Модель не встигла сформувати відповідь за відведений час. "
            "Спробуй коротший запит або повтори спробу."
        )

    if isinstance(error, APIError):
        return f"### Помилка API Ollama\n\n`{error}`"

    if isinstance(error, (ValueError, TypeError, KeyError)):
        return f"### Помилка обробки відповіді\n\n`{type(error).__name__}: {error}`"

    return f"### Непередбачена помилка\n\n`{type(error).__name__}: {error}`"


# ============================================================
# AI-ЧАТ (ФОРМАТ ПОВІДОМЛЕНЬ ДЛЯ GRADIO 6)
# ============================================================

def chat(message, history):
    """Веде діалог і зберігає історію у форматі role/content."""
    history = history or []

    if not message or not message.strip():
        return history, ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Gradio 6 використовує список словників, а не список пар.
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")

            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message.strip()})

    try:
        answer = ask_model(messages)
    except Exception as error:
        print("CHAT ERROR:", repr(error))
        answer = handle_error(error)

    updated_history = history + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": answer}
    ]

    return updated_history, ""


# ============================================================
# ПЕРЕВІРКА СИНТАКСИСУ PYTHON
# ============================================================

def check_python_syntax(code):
    """Перевіряє синтаксис Python вбудованим парсером."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as error:
        line = error.lineno if error.lineno is not None else "невідомий"
        column = error.offset if error.offset is not None else "невідома"
        return (
            "### Виявлено синтаксичну помилку Python\n\n"
            f"**Рядок:** {line}  \n"
            f"**Позиція:** {column}  \n"
            f"**Причина:** `{error.msg}`\n\n"
            "Перевір дужки, лапки, двокрапки та відступи у вказаному рядку."
        )
    except Exception as error:
        return (
            "### Не вдалося перевірити синтаксис\n\n"
            f"`{type(error).__name__}: {error}`"
        )


# ============================================================
# АНАЛІЗАТОР КОДУ
# ============================================================

def analyze_code(code, language, action):
    """Знаходить помилки, виправляє, пояснює або оптимізує код."""
    try:
        if not isinstance(code, str) or not code.strip():
            return "Спочатку встав код для аналізу."

        if not language:
            return "Вибери мову програмування."

        prompts = {
            "Знайти помилки": (
                "Перевір код на синтаксичні та логічні помилки. "
                "Назви конкретні рядки, поясни кожну знайдену помилку "
                "та запропонуй спосіб виправлення. Не вигадуй помилок."
            ),
            "Виправити код": (
                "Виправ помилки у наданому коді. Спочатку покажи повну "
                "виправлену версію у відповідному блоці коду. Потім "
                "коротко перелічи, що саме було змінено. Не змінюй "
                "призначення програми."
            ),
            "Пояснити код": (
                "Поясни код простими словами українською. Розбери "
                "основні частини послідовно та наведи приклад того, "
                "як він працює. Не вигадуй помилок."
            ),
            "Оптимізувати код": (
                "Запропонуй оптимізовану версію коду. Збережи його "
                "початкову функціональність. Покажи повний код і "
                "поясни основні покращення."
            )
        }

        if action not in prompts:
            return f"Невідома дія: {action}"

        # Для Python перевіряємо синтаксис перед роботою LLM.
        syntax_result = None

        if language == "Python" and action in [
            "Знайти помилки",
            "Виправити код"
        ]:
            syntax_result = check_python_syntax(code)

            if syntax_result and action == "Знайти помилки":
                return syntax_result

        prompt = f"""
Мова програмування: {language}

Завдання:
{prompts[action]}

Результат автоматичної перевірки синтаксису:
{syntax_result if syntax_result else "Синтаксичних помилок не виявлено."}

ВАЖЛИВО:
- Відповідай українською мовою.
- Обов'язково враховуй результат автоматичної перевірки.
- Якщо знайдено синтаксичну помилку, не стверджуй, що код правильний.
- У режимі виправлення обов'язково виправ усі знайдені синтаксичні помилки.
- Покажи повний виправлений код.
- Не вигадуй помилок, яких немає у коді.

Код користувача:
```{language.lower()}
{code}
```
"""

        result = ask_model([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])

        return result

    except Exception as error:
        print("ANALYZE_CODE ERROR:", repr(error))
        return handle_error(error)


# ============================================================
# ГЕНЕРАТОР КОДУ
# ============================================================

def generate_code(description, language):
    """Генерує програмний код за текстовим описом."""
    try:
        if not description or not description.strip():
            return "Опиши, яку програму потрібно створити."

        if not language:
            return "Вибери мову програмування."

        prompt = f"""
Мова програмування: {language}

Створи програму за таким описом:
{description.strip()}

Вимоги:
1. Напиши повний приклад коду.
2. Додай доречні коментарі.
3. Після коду поясни принцип його роботи.
4. Якщо потрібні сторонні бібліотеки, вкажи команди їх встановлення.
5. Не стверджуй, що код був виконаний або протестований, якщо цього не робилося.
"""

        return ask_model([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])

    except Exception as error:
        print("GENERATE_CODE ERROR:", repr(error))
        return handle_error(error)


# ============================================================
# ОФОРМЛЕННЯ ІНТЕРФЕЙСУ
# ============================================================

CSS = """
.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    margin-bottom: 5px;
}

#subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 25px;
}

#sidebar {
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 18px;
}

#workspace {
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 20px;
    min-height: 600px;
}

#chatbot {
    border-radius: 12px;
}

#result {
    min-height: 300px;
    border-radius: 12px;
}

button {
    border-radius: 9px !important;
}
"""


# ============================================================
# GRADIO 6.31.0 — ГОЛОВНИЙ ІНТЕРФЕЙС
# ============================================================

with gr.Blocks(
    title="AI Code Assistant",
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate"
    ),
    css=CSS
) as app:

    gr.Markdown("# AI Code Assistant", elem_id="main-title")

    gr.Markdown(
        "Твій локальний помічник із програмування на базі Qwen3",
        elem_id="subtitle"
    )

    with gr.Row():

        # Бічна панель
        with gr.Column(scale=1, min_width=220):
            with gr.Column(elem_id="sidebar"):
                gr.Markdown("### Навігація")

                page = gr.Radio(
                    choices=[
                        "AI-чат",
                        "Аналізатор коду",
                        "Генератор коду"
                    ],
                    value="AI-чат",
                    label="Інструменти"
                )

                gr.Markdown("---")
                gr.Markdown("### Модель")

                gr.Markdown(
                    f"**{MODEL}**\n\n"
                    f"Temperature: `{TEMPERATURE}`\n\n"
                    f"Max tokens: `{MAX_TOKENS}`\n\n"
                    f"Context: `{CONTEXT_LENGTH}`"
                )

                gr.Markdown("---")
                gr.Markdown(
                    "Локальна генерація\n\n"
                    "Сервер: Ollama\n\n"
                    "API: OpenAI-compatible"
                )

        # Робоча область
        with gr.Column(scale=4):

            # ------------------------------------------------
            # СТОРІНКА AI-ЧАТУ
            # ------------------------------------------------
            with gr.Column(
                visible=True,
                elem_id="workspace"
            ) as chat_page:

                gr.Markdown("## AI-чат")

                gr.Markdown(
                    "Став запитання, вивчай програмування "
                    "та отримуй пояснення."
                )

                chatbot = gr.Chatbot(
                    height=430,
                    elem_id="chatbot"
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Напиши своє запитання...",
                        lines=2,
                        scale=5,
                        show_label=False
                    )

                    send_button = gr.Button(
                        "Надіслати",
                        variant="primary",
                        scale=1
                    )

                clear_button = gr.Button(
                    "Очистити історію",
                    variant="secondary"
                )

                send_button.click(
                    fn=chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                )

                chat_input.submit(
                    fn=chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input]
                )

                clear_button.click(
                    fn=lambda: ([], ""),
                    inputs=None,
                    outputs=[chatbot, chat_input]
                )

            # ------------------------------------------------
            # СТОРІНКА АНАЛІЗАТОРА КОДУ
            # ------------------------------------------------
            with gr.Column(
                visible=False,
                elem_id="workspace"
            ) as analyzer_page:

                gr.Markdown("## Аналізатор коду")

                gr.Markdown(
                    "Встав код, вибери мову та дію, "
                    "яку повинен виконати AI."
                )

                language_analyzer = gr.Dropdown(
                    choices=[
                        "Python",
                        "Java",
                        "C#",
                        "C++",
                        "JavaScript",
                        "SQL",
                        "PHP"
                    ],
                    value="Python",
                    label="Мова програмування"
                )

                code_input = gr.Code(
                    language="python",
                    label="Твій код",
                    lines=12
                )

                action = gr.Radio(
                    choices=[
                        "Знайти помилки",
                        "Виправити код",
                        "Пояснити код",
                        "Оптимізувати код"
                    ],
                    value="Знайти помилки",
                    label="Що зробити?"
                )

                analyze_button = gr.Button(
                    "Проаналізувати код",
                    variant="primary"
                )

                analysis_result = gr.Markdown(
                    elem_id="result"
                )

                analyze_button.click(
                    fn=analyze_code,
                    inputs=[
                        code_input,
                        language_analyzer,
                        action
                    ],
                    outputs=analysis_result
                )

                def update_code_language(lang):
                    language_map = {
                        "Python": "python",
                        "Java": "java",
                        "C#": "csharp",
                        "C++": "cpp",
                        "JavaScript": "javascript",
                        "SQL": "sql",
                        "PHP": "php"
                    }
                    return gr.Code(
                        language=language_map.get(lang, "python")
                    )

                language_analyzer.change(
                    fn=update_code_language,
                    inputs=language_analyzer,
                    outputs=code_input
                )

            # ------------------------------------------------
            # СТОРІНКА ГЕНЕРАТОРА КОДУ
            # ------------------------------------------------
            with gr.Column(
                visible=False,
                elem_id="workspace"
            ) as generator_page:

                gr.Markdown("## Генератор коду")

                gr.Markdown(
                    "Опиши завдання, а AI створить програму."
                )

                language_generator = gr.Dropdown(
                    choices=[
                        "Python",
                        "Java",
                        "C#",
                        "C++",
                        "JavaScript",
                        "SQL",
                        "PHP"
                    ],
                    value="Python",
                    label="Мова програмування"
                )

                description_input = gr.Textbox(
                    label="Опис завдання",
                    placeholder=(
                        "Наприклад: створи програму, "
                        "яка сортує список чисел..."
                    ),
                    lines=7
                )

                generate_button = gr.Button(
                    "Згенерувати код",
                    variant="primary"
                )

                generation_result = gr.Markdown(
                    elem_id="result"
                )

                generate_button.click(
                    fn=generate_code,
                    inputs=[
                        description_input,
                        language_generator
                    ],
                    outputs=generation_result
                )

            # ------------------------------------------------
            # ПЕРЕМИКАННЯ МІЖ СТОРІНКАМИ
            # ------------------------------------------------
            def change_page(selected):
                return (
                    gr.update(visible=selected == "AI-чат"),
                    gr.update(visible=selected == "Аналізатор коду"),
                    gr.update(visible=selected == "Генератор коду")
                )

            page.change(
                fn=change_page,
                inputs=page,
                outputs=[
                    chat_page,
                    analyzer_page,
                    generator_page
                ]
            )

    gr.Markdown(
        "---\n"
        "AI Code Assistant | Ollama + Qwen3 1.7B | "
        "Локальний запуск"
    )


if __name__ == "__main__":
    app.queue().launch()
