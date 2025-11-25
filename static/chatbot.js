// static/chatbot.js

const chatbotToggle = document.getElementById("chatbotToggle");
const chatbotWidget = document.getElementById("chatbotWidget");
const closeChatbot = document.getElementById("closeChatbot");
const chatbotForm = document.getElementById("chatbotForm");
const chatbotInput = document.getElementById("chatbotInput");
const chatbotMessages = document.getElementById("chatbotMessages");

function addMessage(text, sender = "bot") {
  if (!chatbotMessages) return;
  const wrapper = document.createElement("div");
  wrapper.classList.add("chatbot-message", sender);
  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  chatbotMessages.appendChild(wrapper);
  chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
}

function generateBotReply(userMessage) {
  const msg = userMessage.toLowerCase();

  if (msg.includes("hello") || msg.includes("hi")) {
    return "Hello! I'm StuBot 🤖. Ask me about study tips, attendance, or how to improve your performance.";
  }
  if (msg.includes("attendance")) {
    return "Try to maintain at least 75% attendance. It keeps you in touch with the subject and reduces last-minute stress.";
  }
  if (msg.includes("study") && msg.includes("hours")) {
    return "For most B.Tech subjects, 2–3 hours of focused self-study per day is a good target. Use the Pomodoro technique and short breaks.";
  }
  if (msg.includes("weak") || msg.includes("low marks")) {
    return "Start from your weakest subject, use standard textbooks, solve PYQs and revise weekly instead of cramming at the end.";
  }
  if (msg.includes("gate")) {
    return "For GATE: focus on core subjects, solve previous-year questions, give mock tests, and maintain short formula notes.";
  }
  if (msg.includes("marks") && msg.includes("improve")) {
    return "Analyse where you lose marks: concepts, silly mistakes or time. Fix them with daily practice and revision.";
  }
  if (msg.includes("model") || msg.includes("predict")) {
    return "The website uses Python models like Linear Regression and Random Forest trained with NumPy, pandas and scikit-learn.";
  }
  return "I'm a simple chatbot. Try asking about attendance, study hours, weak students, improvement tips or GATE preparation 😊.";
}

if (chatbotToggle && chatbotWidget) {
  chatbotToggle.addEventListener("click", () => {
    chatbotWidget.classList.toggle("hidden");
    if (!chatbotWidget.classList.contains("hidden") && chatbotInput) {
      chatbotInput.focus();
    }
  });
}

if (closeChatbot && chatbotWidget) {
  closeChatbot.addEventListener("click", () => {
    chatbotWidget.classList.add("hidden");
  });
}

if (chatbotForm && chatbotInput) {
  chatbotForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = chatbotInput.value.trim();
    if (!text) return;
    addMessage(text, "user");
    const reply = generateBotReply(text);
    setTimeout(() => addMessage(reply, "bot"), 300);
    chatbotInput.value = "";
  });
}

// initial greeting
if (chatbotMessages && chatbotMessages.children.length === 0) {
  addMessage("Hi, I'm StuBot 🤖. I live in this bubble and can give you basic study suggestions.");
}


