from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Telegram Bot Token
TOKEN = '7855321166:AAEdpAMfSxUQKxRk_DauAEpVM2YQ4YEi2uM'


async def start(update: Update, context):
    await update.message.reply_text(
        "Hello! I am your AI assistant. Ask me anything, and I will try my best to answer!"
    )

# Define the message handler to process user input
async def handle_message(update: Update, context):
    user_input = update.message.text.strip()  # Preprocess user input

    
    try:
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=100,       
            temperature=0.7,      
            top_k=50,            
            top_p=0.9             
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Fallback for empty or nonsensical responses
        if not response or len(response.split()) < 3:
            response = (
                "I'm sorry, I couldn't understand your question. Could you please rephrase?"
            )

    except Exception as e:
        response = (
            "Oops! Something went wrong while processing your request. Please try again later."
        )
        print(f"Error: {e}")  

    # Send response back to user
    await update.message.reply_text(response)

# Set up the Telegram bot application
app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Run the bot
if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()
