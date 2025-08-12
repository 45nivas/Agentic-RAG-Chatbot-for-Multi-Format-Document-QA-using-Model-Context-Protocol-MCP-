// Global variables
let uploadedFiles = [];
let isProcessing = false;
let recognition = null;
let isRecording = false;

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadedFilesSection = document.getElementById('uploadedFiles');
const fileList = document.getElementById('fileList');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const micIcon = document.getElementById('micIcon');
const micSpinner = document.getElementById('micSpinner');
const voiceStatus = document.getElementById('voiceStatus');
const voiceStatusText = document.getElementById('voiceStatusText');
const clearBtn = document.getElementById('clearBtn');
const chatContainer = document.getElementById('chatContainer');
const statusModal = document.getElementById('statusModal');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const modalSpinner = document.getElementById('modalSpinner');

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeSpeechRecognition();
    checkExistingFiles();
});

// Event listeners
function initializeEventListeners() {
    // File upload events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Chat events
    chatInput.addEventListener('keypress', handleChatKeyPress);
    sendBtn.addEventListener('click', sendMessage);
    micBtn.addEventListener('click', toggleVoiceInput);
    clearBtn.addEventListener('click', clearConversation);
    
    // Focus management
    chatInput.addEventListener('focus', scrollToBottom);
}

// File upload handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
}

// Process uploaded files
async function processFiles(files) {
    if (isProcessing) return;
    
    // Validate files
    const validFiles = validateFiles(files);
    if (validFiles.length === 0) return;
    
    isProcessing = true;
    showModal('Uploading Files...', 'Processing your documents, please wait...');
    
    try {
        const formData = new FormData();
        validFiles.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            uploadedFiles = result.files;
            updateUploadedFilesList();
            enableChat();
            showSuccessMessage(`Successfully uploaded ${result.files.length} file(s)!`);
        } else {
            showErrorMessage(result.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showErrorMessage('Upload failed. Please try again.');
    } finally {
        isProcessing = false;
        hideModal();
    }
}

// Validate uploaded files
function validateFiles(files) {
    const validExtensions = ['.pdf', '.docx', '.pptx', '.csv', '.txt', '.md'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    const validFiles = [];
    
    for (const file of files) {
        // Check file extension
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        if (!validExtensions.includes(extension)) {
            showErrorMessage(`Invalid file type: ${file.name}. Supported types: PDF, DOCX, PPTX, CSV, TXT, MD`);
            continue;
        }
        
        // Check file size
        if (file.size > maxSize) {
            showErrorMessage(`File too large: ${file.name}. Maximum size is 16MB.`);
            continue;
        }
        
        validFiles.push(file);
    }
    
    return validFiles;
}

// Update uploaded files display
function updateUploadedFilesList() {
    const uploadStats = document.getElementById('uploadStats');
    const fileCount = document.getElementById('fileCount');
    const chatSubtitle = document.getElementById('chatSubtitle');
    
    if (uploadedFiles.length === 0) {
        uploadedFilesSection.style.display = 'none';
        if (uploadStats) uploadStats.style.display = 'none';
        if (chatSubtitle) chatSubtitle.textContent = 'Upload documents to start asking questions';
        return;
    }
    
    fileList.innerHTML = '';
    uploadedFiles.forEach(filename => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span>📄 ${filename}</span>
        `;
        li.classList.add('fade-in');
        fileList.appendChild(li);
    });
    
    uploadedFilesSection.style.display = 'block';
    
    // Update upload stats
    if (uploadStats && fileCount) {
        fileCount.textContent = uploadedFiles.length;
        uploadStats.style.display = 'block';
    }
    
    // Update chat subtitle
    if (chatSubtitle) {
        chatSubtitle.textContent = `${uploadedFiles.length} document${uploadedFiles.length !== 1 ? 's' : ''} ready for questions`;
    }
}

// Enable chat functionality
function enableChat() {
    chatInput.disabled = false;
    sendBtn.disabled = false;
    micBtn.disabled = false;
    chatInput.placeholder = "Ask a question about your documents...";
    clearBtn.style.display = 'inline-block';
    
    // Clear welcome message and show chat ready message
    const welcomeMessage = chatContainer.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }
    
    addMessage('bot', `Great! Your documents have been processed. I can help you find information from your uploaded files.

💡 **Quick tips for better results:**
• Be specific in your questions
• Ask about particular topics or sections
• Use the microphone 🎤 for voice input

What would you like to know about your documents?`);
}

// Chat handlers
function handleChatKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isProcessing) return;
    
    // Add user message
    addMessage('user', message);
    chatInput.value = '';
    
    // Show input status and typing indicator
    showInputStatus();
    const typingId = addTypingIndicator();
    
    isProcessing = true;
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const result = await response.json();
        
        // Remove typing indicator and input status
        removeTypingIndicator(typingId);
        hideInputStatus();
        
        if (result.success) {
            addMessage('bot', result.response, result.source_context);
            
            // Add debug info for developers (only show if similarity is low)
            if (result.metadata && result.metadata.max_similarity < 0.6) {
                console.log('Debug Info:', {
                    similarity: result.metadata.max_similarity,
                    threshold_met: result.metadata.threshold_met,
                    chunks_found: result.metadata.chunks_found
                });
                
                // Show a subtle debug indicator
                const lastMessage = chatContainer.lastElementChild;
                if (lastMessage) {
                    const debugInfo = document.createElement('div');
                    debugInfo.className = 'debug-info';
                    debugInfo.innerHTML = `<small>🔍 Similarity: ${(result.metadata.max_similarity * 100).toFixed(1)}%</small>`;
                    lastMessage.querySelector('.message-content').appendChild(debugInfo);
                }
            }
        } else {
            addMessage('bot', result.error || 'Sorry, I encountered an error while processing your question.');
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        hideInputStatus();
        addMessage('bot', 'Sorry, I encountered a network error. Please try again.');
    } finally {
        isProcessing = false;
    }
}

// Add message to chat
function addMessage(sender, content, sourceContext = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender, 'fade-in');
    
    const avatar = document.createElement('div');
    avatar.classList.add('message-avatar');
    avatar.textContent = sender === 'user' ? 'U' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    
    // Format content (basic markdown support)
    const formattedContent = formatMessage(content);
    contentDiv.innerHTML = formattedContent;
    
    // Add helpful suggestions for low-relevance responses
    if (sender === 'bot' && content.includes('more specific to the document content')) {
        const suggestionsDiv = document.createElement('div');
        suggestionsDiv.classList.add('suggestions-box');
        
        // Detect if this might be a resume based on the uploaded filenames
        const resumeKeywords = ['resume', 'cv', 'lakshmi', 'nivas', 'ppo'];
        const isLikelyResume = uploadedFiles.some(filename => 
            resumeKeywords.some(keyword => filename.toLowerCase().includes(keyword))
        );
        
        if (isLikelyResume) {
            suggestionsDiv.innerHTML = `
                <div class="suggestions-header">💡 For resume/CV documents, try asking:</div>
                <button class="suggestion-btn" onclick="fillInputAndSend('What are the technical skills mentioned?')">Technical skills</button>
                <button class="suggestion-btn" onclick="fillInputAndSend('What is the educational background?')">Education background</button>
                <button class="suggestion-btn" onclick="fillInputAndSend('What work experience is listed?')">Work experience</button>
                <button class="suggestion-btn" onclick="fillInputAndSend('What programming languages are mentioned?')">Programming languages</button>
                <button class="suggestion-btn" onclick="fillInputAndSend('What AI/ML skills are listed?')">AI/ML skills</button>
            `;
        } else {
            suggestionsDiv.innerHTML = `
                <div class="suggestions-header">💡 Try these example questions:</div>
                <button class="suggestion-btn" onclick="fillInputAndSend('What are the main topics in this document?')">What are the main topics?</button>
                <button class="suggestion-btn" onclick="fillInputAndSend('Can you provide a summary?')">Provide a summary</button>
                <button class="suggestion-btn" onclick="fillInputAndSend('What are the key findings?')">Key findings</button>
                <button class="suggestion-btn" onclick="fillInputAndSend('What conclusions are mentioned?')">Conclusions</button>
            `;
        }
        contentDiv.appendChild(suggestionsDiv);
    }
    
    // Add source context if available (for bot messages)
    if (sender === 'bot' && sourceContext && sourceContext.length > 0) {
        const sourceDiv = document.createElement('div');
        sourceDiv.classList.add('source-context');
        sourceDiv.innerHTML = `
            <details class="source-details">
                <summary class="source-summary">📋 View Source Context (${sourceContext.length} chunks)</summary>
                <div class="source-list">
                    ${sourceContext.map((context, index) => `
                        <div class="source-item">
                            <strong>Chunk ${index + 1}:</strong> 
                            <span class="source-text">${context.length > 200 ? context.substring(0, 200) + '...' : context}</span>
                        </div>
                    `).join('')}
                </div>
            </details>
        `;
        contentDiv.appendChild(sourceDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    // Remove welcome message if it exists
    const welcomeMessage = chatContainer.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

// New function to fill input and send message
function fillInputAndSend(message) {
    chatInput.value = message;
    sendMessage();
}

// Add typing indicator
function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'bot', 'typing-indicator');
    
    const avatar = document.createElement('div');
    avatar.classList.add('message-avatar');
    avatar.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.innerHTML = `
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(contentDiv);
    chatContainer.appendChild(typingDiv);
    scrollToBottom();
    
    return typingDiv;
}

// Remove typing indicator
function removeTypingIndicator(typingElement) {
    if (typingElement && typingElement.parentNode) {
        typingElement.parentNode.removeChild(typingElement);
    }
}

// Format message content
function formatMessage(content) {
    // Basic markdown formatting
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

// Clear conversation
async function clearConversation() {
    if (isProcessing) return;
    
    if (!confirm('Are you sure you want to clear the conversation?')) {
        return;
    }
    
    try {
        const response = await fetch('/clear', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            chatContainer.innerHTML = '';
            addMessage('bot', 'Conversation cleared! How can I help you with your documents?');
        } else {
            showErrorMessage('Failed to clear conversation');
        }
    } catch (error) {
        console.error('Clear error:', error);
        showErrorMessage('Failed to clear conversation');
    }
}

// Check for existing files on page load
async function checkExistingFiles() {
    try {
        const response = await fetch('/health');
        const result = await response.json();
        
        if (result.has_files) {
            enableChat();
            // Don't show uploaded files list as we don't have the filenames
            chatContainer.innerHTML = '';
            addMessage('bot', `Welcome back! Your documents are ready. You can ask questions like:
            
• "What are the technical skills mentioned?"
• "What is the educational background?" 
• "What AI/ML experience is listed?"
• "Can you summarize the key qualifications?"

Or use the microphone button 🎤 for voice input! What would you like to know?`);
        }
    } catch (error) {
        console.error('Health check error:', error);
    }
}

// Modal functions
function showModal(title, message) {
    modalTitle.textContent = title;
    modalMessage.textContent = message;
    modalSpinner.style.display = 'flex';
    statusModal.style.display = 'flex';
}

function hideModal() {
    statusModal.style.display = 'none';
}

// Message functions
function showSuccessMessage(message) {
    showToast(message, 'success');
}

function showErrorMessage(message) {
    showToast(message, 'error');
}

function showToast(message, type) {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => toast.remove());
    
    const toast = document.createElement('div');
    toast.classList.add('toast', type, 'slide-up');
    toast.textContent = message;
    
    // Style the toast
    Object.assign(toast.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        borderRadius: '8px',
        color: 'white',
        fontWeight: '500',
        zIndex: '1001',
        maxWidth: '400px',
        wordWrap: 'break-word',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        background: type === 'success' ? 
            'linear-gradient(135deg, #10b981, #059669)' : 
            'linear-gradient(135deg, #ef4444, #dc2626)'
    });
    
    document.body.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
    
    // Click to dismiss
    toast.addEventListener('click', () => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    });
}

// Utility functions
function scrollToBottom() {
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
}

// Speech Recognition Functions
function initializeSpeechRecognition() {
    // Check for browser support
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        micBtn.style.display = 'none';
        return;
    }

    // Initialize Speech Recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    // Configure recognition settings
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    // Recognition event handlers
    recognition.onstart = function() {
        isRecording = true;
        micBtn.classList.add('recording');
        micIcon.style.display = 'none';
        micSpinner.style.display = 'flex';
        showVoiceStatus('Listening... Speak now', false);
    };
    
    recognition.onresult = function(event) {
        let finalTranscript = '';
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Update input with transcript
        if (finalTranscript) {
            chatInput.value = finalTranscript;
            showVoiceStatus('✅ Voice input completed! Click send or press Enter', false);
            setTimeout(hideVoiceStatus, 3000);
            // Auto-focus the input for immediate editing if needed
            chatInput.focus();
        } else if (interimTranscript) {
            chatInput.value = interimTranscript;
            showVoiceStatus('🎤 Listening... "' + interimTranscript + '"', false);
        }
    };
    
    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
        let errorMessage = 'Voice input failed';
        
        switch(event.error) {
            case 'no-speech':
                errorMessage = 'No speech detected. Please try again.';
                break;
            case 'audio-capture':
                errorMessage = 'Microphone not available.';
                break;
            case 'not-allowed':
                errorMessage = 'Microphone permission denied.';
                break;
            case 'network':
                errorMessage = 'Network error. Please check your connection.';
                break;
            default:
                errorMessage = 'Voice input error: ' + event.error;
        }
        
        showVoiceStatus(errorMessage, true);
        setTimeout(hideVoiceStatus, 3000);
        stopRecording();
    };
    
    recognition.onend = function() {
        stopRecording();
    };
}

function toggleVoiceInput() {
    if (isRecording) {
        recognition.stop();
    } else {
        if (recognition) {
            recognition.start();
        } else {
            showVoiceStatus('Speech recognition not supported in this browser', true);
            setTimeout(hideVoiceStatus, 3000);
        }
    }
}

function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    micIcon.style.display = 'block';
    micSpinner.style.display = 'none';
}

function showVoiceStatus(message, isError = false) {
    voiceStatusText.textContent = message;
    voiceStatus.className = 'voice-status' + (isError ? ' error' : '');
    voiceStatus.style.display = 'block';
}

function hideVoiceStatus() {
    voiceStatus.style.display = 'none';
}

// Add CSS for typing dots animation
const style = document.createElement('style');
style.textContent = `
    .typing-dots {
        display: flex;
        gap: 4px;
        align-items: center;
    }
    
    .typing-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #94a3b8;
        animation: typing 1.5s infinite;
    }
    
    .typing-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: scale(1);
            opacity: 0.5;
        }
        30% {
            transform: scale(1.2);
            opacity: 1;
        }
    }
    
    .toast {
        transition: all 0.3s ease;
    }
`;
document.head.appendChild(style);

// Input status helpers for better UX
function showInputStatus() {
    const inputStatus = document.getElementById('inputStatus');
    if (inputStatus) {
        inputStatus.style.display = 'flex';
    }
}

function hideInputStatus() {
    const inputStatus = document.getElementById('inputStatus');
    if (inputStatus) {
        inputStatus.style.display = 'none';
    }
}
