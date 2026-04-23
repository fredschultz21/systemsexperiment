// @ts-nocheck
import { useState, useRef, useEffect } from "react";

interface Message {
    role: "user" | "ai";
    text: string;
    displayed?: string;
    image?: string;
}

function Home() {
    const [isPrompted, setIsPrompted] = useState(false);
    const [inputValue, setInputValue] = useState("");
    const [messages, setMessages] = useState<Message[]>([]);
    const [imagePreview, setImagePreview] = useState(null);
    const [imageB64, setImageB64] = useState(null);
    const bottomRef = useRef<HTMLDivElement>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const isAtBottomRef = useRef(true);
    const fileInputRef = useRef(null);

    const scrollToBottom = (smooth = true) => {
        if (!isAtBottomRef.current) return;
        bottomRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "instant" });
    };

    const handleScroll = () => {
        const el = scrollRef.current;
        if (!el) return;
        const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        isAtBottomRef.current = distanceFromBottom < 60;
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result.split(",")[1];
            setImageB64(base64);
            setImagePreview(reader.result);
        };
        reader.readAsDataURL(file);
    };

    const clearImage = () => {
        setImageB64(null);
        setImagePreview(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const displayWordByWord = (text: string, messageIndex: number) => {
        const words = text.split(" ");
        let i = 0;
        const interval = setInterval(() => {
            if (i >= words.length) {
                clearInterval(interval);
                return;
            }
            setMessages(prev =>
                prev.map((msg, idx) =>
                    idx === messageIndex
                        ? { ...msg, displayed: words.slice(0, i + 1).join(" ") }
                        : msg
                )
            );
            i++;
        }, 60);
    };

    const handleSubmit = async (prompt: string) => {
        if (!prompt.trim()) return;

        const aiMessageIndex = messages.length + 1;

        setMessages(prev => [
            ...prev,
            { role: "user", text: prompt, image: imagePreview },
            { role: "ai", text: "", displayed: "" }
        ]);
        setInputValue("");
        setIsPrompted(true);
        isAtBottomRef.current = true;

        const currentImageB64 = imageB64;
        clearImage();

        try {
            const body: any = { message: prompt };
            if (currentImageB64) body.image = currentImageB64;

            const response = await fetch(`${import.meta.env.VITE_API_URL}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });

            const data = await response.json();
            const aiText = data.choices[0].message.content;

            setMessages(prev =>
                prev.map((msg, idx) =>
                    idx === aiMessageIndex
                        ? { ...msg, text: aiText }
                        : msg
                )
            );

            displayWordByWord(aiText, aiMessageIndex);
        } catch (err) {
            setMessages(prev =>
                prev.map((msg, idx) =>
                    idx === aiMessageIndex
                        ? { ...msg, text: "Something went wrong.", displayed: "Something went wrong." }
                        : msg
                )
            );
        }
    };

    const inputEl = (
        <div className="flex flex-col items-center gap-2 w-1/3">
            {imagePreview && (
                <div className="relative">
                    <img src={imagePreview} className="h-16 w-16 object-cover rounded-lg" />
                    <button
                        onClick={clearImage}
                        className="absolute -top-1 -right-1 bg-stone-600 text-white rounded-full w-4 h-4 text-xs flex items-center justify-center"
                    >×</button>
                </div>
            )}
            <div className="flex w-full gap-2">
                <input
                    type="text"
                    value={inputValue}
                    placeholder="How can I help you today?"
                    className="bg-stone-800 flex-1 px-5 py-3 text-stone-200 rounded-3xl transition-colors duration-200 placeholder-stone-500 font-serif border outline-none border-stone-800 focus:border-stone-600 caret-stone-300"
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter" && inputValue.trim() !== "") {
                            handleSubmit(inputValue);
                        }
                    }}
                />
                <button
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-stone-700 hover:bg-stone-600 text-stone-300 px-3 py-3 rounded-full transition-colors"
                    title="Upload image"
                >
                    📷
                </button>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleImageChange}
                />
            </div>
            <p className="text-stone-500 text-xs font-serif">
                This is general guidance, not certified financial advice.
            </p>
        </div>
    );

    return (
        <>
            {!isPrompted ? (
                <div className="flex flex-col bg-stone-900 h-screen">
                    <div className="flex items-center justify-center flex-1 flex-col pb-32">
                        <p className="text-4xl text-red-50 font-serif py-8">
                            Here for your consultation needs
                        </p>
                        {inputEl}
                    </div>
                </div>
            ) : (
                <div className="flex flex-col bg-stone-900 h-screen">
                    <div
                        ref={scrollRef}
                        onScroll={handleScroll}
                        className="flex-1 overflow-y-auto scrollbar-hide px-4 py-10"
                        style={{ scrollBehavior: "smooth" }}
                    >
                        <div className="w-1/2 mx-auto flex flex-col gap-4">
                            {messages.map((msg, idx) => (
                                <div
                                    key={idx}
                                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                                >
                                    <div
                                        className={`px-4 py-3 rounded-2xl max-w-[75%] font-serif text-base leading-relaxed ${msg.role === "user"
                                            ? "bg-stone-600 text-red-50 rounded-br-sm"
                                            : "bg-stone-800 text-red-50 rounded-bl-sm"
                                            }`}
                                    >
                                        {msg.image && (
                                            <img src={msg.image} className="max-h-40 rounded-lg mb-2 object-cover" />
                                        )}
                                        {msg.role === "ai" && msg.displayed === "" ? (
                                            <span className="flex gap-1 items-center h-5">
                                                <span className="w-2 h-2 bg-stone-400 rounded-full animate-bounce [animation-delay:0ms]" />
                                                <span className="w-2 h-2 bg-stone-400 rounded-full animate-bounce [animation-delay:150ms]" />
                                                <span className="w-2 h-2 bg-stone-400 rounded-full animate-bounce [animation-delay:300ms]" />
                                            </span>
                                        ) : (
                                            <>
                                                {msg.role === "ai" ? msg.displayed : msg.text}
                                                {msg.role === "ai" && msg.displayed && msg.displayed !== msg.text && (
                                                    <span className="inline-block w-1 h-4 ml-1 bg-stone-400 animate-pulse align-middle" />
                                                )}
                                            </>
                                        )}
                                    </div>
                                </div>
                            ))}
                            <div ref={bottomRef} />
                        </div>
                    </div>
                    <div className="flex items-center justify-center py-6">
                        {inputEl}
                    </div>
                </div>
            )}
        </>
    );
}

export default Home;