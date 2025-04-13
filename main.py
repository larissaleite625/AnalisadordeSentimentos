import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, KeywordsOptions, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from tkhtmlview import HTMLLabel
import markdown

load_dotenv()

directory_path = r"C:\Users\PC\Desktop\Projetos\AnaliseSentimentos"

class SentimentAnalyzerApp(tk.Tk):
    def toggle_theme(self):
        if self.theme == "light":
            self.theme = "dark"
            self.theme_button.config(text="Tema: Escuro")
            self.configure(bg="#2e2e2e")
            self.insights_html.configure(background="#2e2e2e", foreground="white")
        else:
            self.theme = "light"
            self.theme_button.config(text="Tema: Claro")
            self.configure(bg="white")
            self.insights_html.configure(background="white", foreground="black")

    def __init__(self):
        super().__init__()
        self.title("Analisador de Sentimentos")
        self.geometry("800x600")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.running = False
        self.theme = "light"

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.theme_button = ttk.Button(main_frame, text="Tema: Claro", command=self.toggle_theme)
        self.theme_button.pack(anchor=tk.NE)

        ttk.Label(main_frame, text="Digite o texto para análise:").pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=10)
        self.input_text.pack(fill=tk.X, pady=5)

        self.analyze_button = ttk.Button(main_frame, text="Analisar", command=self.start_analysis)
        self.analyze_button.pack(pady=5)

        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')

        result_frame = ttk.LabelFrame(main_frame, text="Resultados")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.sentiment_label = ttk.Label(result_frame, text="Sentimento predominante: ")
        self.sentiment_label.pack(anchor=tk.W)

        ttk.Label(result_frame, text="Insights:").pack(anchor=tk.W)
        self.insights_html = HTMLLabel(result_frame, html="", background="white")
        self.insights_html.pack(fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.pack()

    def start_analysis(self):
        if not self.running:
            self.running = True
            input_text = self.input_text.get("1.0", tk.END).strip()

            if not input_text:
                messagebox.showwarning("Entrada vazia", "Por favor digite um texto para análise.")
                self.running = False
                return

            self.analyze_button.config(state=tk.DISABLED)
            self.progress.pack()
            self.progress.start()
            self.status_label.config(text="Analisando...")

            analysis_thread = threading.Thread(target=self.run_analysis, args=(input_text,))
            analysis_thread.start()

    def run_analysis(self, input_text):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = create_directory(timestamp)

            response_ibm = analyze_sentiment(input_text, directory)
            sentiment = response_ibm['sentiment']['document']
            sentiment_text = f"Sentimento predominante: {sentiment['label'].capitalize()}, Score: {sentiment['score']:.2f}"

            insights = analyze_insights_with_deepseek(directory, input_text)

            self.after(0, self.update_results, sentiment_text, insights)

        except Exception as e:
            self.after(0, messagebox.showerror, "Erro", f"Ocorreu um erro:\n{str(e)}")

        finally:
            self.after(0, self.analysis_complete)

    def update_results(self, sentiment, insights_md):
        self.sentiment_label.config(text=sentiment)
        html_content = markdown.markdown(insights_md)
        self.insights_html.set_html(html_content)

    def analysis_complete(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.analyze_button.config(state=tk.NORMAL)
        self.status_label.config(text="Análise concluída!")
        self.running = False

    def on_close(self):
        if messagebox.askokcancel("Sair", "Deseja realmente fechar o programa?"):
            self.destroy()

def create_directory(timestamp):
    directory_name = f"analysis_{timestamp}"
    os.makedirs(directory_name, exist_ok=True)
    return directory_name

def analyze_sentiment(input_text, directory):
    authenticator = IAMAuthenticator(os.getenv("IBM_API_KEY"))
    nlu = NaturalLanguageUnderstandingV1(
        version='2022-04-07',
        authenticator=authenticator
    )
    nlu.set_service_url(os.getenv("IBM_API_URL"))

    response = nlu.analyze(
        text=input_text,
        features=Features(
            sentiment=SentimentOptions(),
            keywords=KeywordsOptions(sentiment=True, limit=5),
            emotion=EmotionOptions()
        )
    ).get_result()

    with open(os.path.join(directory, "responsefromibmwatson.txt"), 'w', encoding='utf-8') as file:
        file.write(json.dumps(response, ensure_ascii=False, indent=4))

    return response

def analyze_insights_with_deepseek(directory, input_text):
    with open(os.path.join(directory, "responsefromibmwatson.txt"), 'r', encoding='utf-8') as file:
        combined_text = file.read()

    with open(os.path.join(directory_path, "Sentimentos.txt"), 'r', encoding='utf-8') as file:
        sentimentos = file.read()

    request_deepseek = {
        "prompt": f"Analise e forneça insights baseados nos seguintes resultados:\n{combined_text} do texto original: {input_text}\nCom os dados adicionais: {sentimentos}\n\n"
                  "Responda com uma análise detalhada, incluindo sentimentos, emoções e palavras-chave relevantes. "
                  "Se houver sentimentos negativos, forneça sugestões de como melhorar a situação. "
    }

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": request_deepseek["prompt"]}],
        stream=False
    )

    insights = response.choices[0].message.content

    with open(os.path.join(directory, "responsefromdeepseek.txt"), 'w', encoding='utf-8') as file:
        file.write(insights)

    return insights

if __name__ == "__main__":
    app = SentimentAnalyzerApp()
    app.mainloop()
