// Global variables
let uploadedFiles = [];
let isProcessing = false;

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadedFilesSection = document.getElementById('uploadedFiles');
const fileList = document.getElementById('fileList');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const chatContainer = document.getElementById('chatContainer');
const statusModal = document.getElementById('statusModal');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const modalSpinner = document.getElementById('modalSpinner');

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
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
    if (uploadedFiles.length === 0) {
        uploadedFilesSection.style.display = 'none';
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
}

// Enable chat functionality
function enableChat() {
    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.placeholder = "Ask a question about your documents...";
    clearBtn.style.display = 'inline-block';
    
    // Clear welcome message and show chat ready message
    const welcomeMessage = chatContainer.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }
    
    addMessage('bot', 'Great! Your documents have been processed. You can now ask me questions about them. What would you like to know?');
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
    
    // Show typing indicator
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
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (result.success) {
            addMessage('bot', result.response, result.source_context);
        } else {
            addMessage('bot', result.error || 'Sorry, I encountered an error while processing your question.');
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
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
            addMessage('bot', 'Welcome back! Your documents are ready. What would you like to know?');
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
