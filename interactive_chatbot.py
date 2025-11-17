"""
✅ INTERACTIVE CHATBOT INTERFACE FOR MEDICAL RAG SYSTEM
Apollo 2B + Medical RAG with Jupyter Widgets (ChatGPT-like UI)

Date: 2025-11-16
Features:
- Real-time chat interface
- Message history display
- Typing indicators
- User feedback system
- Medical answer with metadata
- Beautiful UI with gradients
- Copy buttons
- Clear history
- Export chat
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output, Markdown
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import uuid

# ============================================================================
# INTERACTIVE CHATBOT INTERFACE
# ============================================================================

class InteractiveMedicalChatbot:
    """Interactive ChatGPT-like interface for Medical RAG System"""
    
    def __init__(self, rag_system):
        """
        Initialize chatbot with RAG system
        
        Args:
            rag_system: Initialized AdvancedMedicalRAG instance
        """
        self.rag_system = rag_system
        self.chat_history = []
        self.session_id = str(uuid.uuid4())[:8]
        self.chat_log_file = Path(f"chat_history_{self.session_id}.json")
        
        print("✅ Initializing Interactive Medical Chatbot...")
        self._create_ui()
        print("✅ UI Created. Display with: chatbot.show()")
    
    def _create_ui(self):
        """Create the Jupyter widgets interface"""
        
        # ====== HEADER ======
        self.header = widgets.HTML(
            value="""
            <div style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            '>
                <h1 style='color: white; margin: 0; font-size: 28px;'>
                    🏥 Medical AI Assistant
                </h1>
                <p style='color: #e0e0e0; margin: 5px 0 0 0; font-size: 14px;'>
                    Powered by Apollo 2B + Advanced Medical RAG
                </p>
            </div>
            """
        )
        
        # ====== CHAT DISPLAY AREA ======
        self.chat_display = widgets.HTML(
            value="<div style='text-align: center; color: #999; padding: 20px;'><em>Chat will appear here...</em></div>",
            layout=widgets.Layout(
                height='400px',
                overflow_y='auto',
                border='1px solid #ddd',
                padding='15px',
                border_radius='5px',
                background_color='#f9f9f9'
            )
        )
        
        # ====== INPUT AREA ======
        self.input_text = widgets.Text(
            value='',
            placeholder='Ask a medical question...',
            description='',
            layout=widgets.Layout(width='100%', height='50px'),
            style={'description_width': '0px'}
        )
        self.input_text.on_submit(self._on_submit)
        
        # ====== BUTTONS ======
        self.send_btn = widgets.Button(
            description='📤 Send',
            button_style='info',
            layout=widgets.Layout(width='100px', height='40px'),
            tooltip='Send message (or press Enter)'
        )
        self.send_btn.on_click(self._on_send_click)
        
        self.clear_btn = widgets.Button(
            description='🗑️ Clear',
            button_style='warning',
            layout=widgets.Layout(width='100px', height='40px'),
            tooltip='Clear chat history'
        )
        self.clear_btn.on_click(self._on_clear_click)
        
        self.export_btn = widgets.Button(
            description='💾 Export',
            button_style='success',
            layout=widgets.Layout(width='100px', height='40px'),
            tooltip='Export chat as JSON'
        )
        self.export_btn.on_click(self._on_export_click)
        
        self.settings_btn = widgets.Button(
            description='⚙️ Settings',
            button_style='primary',
            layout=widgets.Layout(width='100px', height='40px'),
            tooltip='Open settings'
        )
        self.settings_btn.on_click(self._on_settings_click)
        
        # ====== SETTINGS PANEL (Hidden by default) ======
        self.use_cot = widgets.Checkbox(
            value=True,
            description='Chain-of-Thought',
            indent=False
        )
        self.use_reranking = widgets.Checkbox(
            value=True,
            description='Use Reranking',
            indent=False
        )
        self.web_search = widgets.Checkbox(
            value=True,
            description='Web Search',
            indent=False
        )
        self.temperature_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            description='Temperature:',
            style={'description_width': '100px'}
        )
        
        self.settings_panel = widgets.VBox([
            widgets.HTML("<b style='font-size: 14px;'>⚙️ Settings</b>"),
            self.use_cot,
            self.use_reranking,
            self.web_search,
            self.temperature_slider,
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='15px',
            border_radius='5px',
            margin='10px 0',
            display='none'
        ))
        
        # ====== STATS PANEL ======
        self.stats_display = widgets.HTML(
            value=self._get_stats_html(),
            layout=widgets.Layout(width='100%')
        )
        
        # ====== STATUS MESSAGE ======
        self.status_message = widgets.HTML(value="")
        
        # ====== LAYOUT ======
        input_row = widgets.HBox([
            self.input_text,
            self.send_btn,
        ], layout=widgets.Layout(width='100%'))
        
        button_row = widgets.HBox([
            self.clear_btn,
            self.export_btn,
            self.settings_btn,
        ], layout=widgets.Layout(width='100%', margin='10px 0'))
        
        self.main_container = widgets.VBox([
            self.header,
            self.chat_display,
            input_row,
            button_row,
            self.settings_panel,
            self.stats_display,
            self.status_message,
        ])
    
    def _on_submit(self, sender):
        """Handle Enter key in input"""
        self._process_input()
    
    def _on_send_click(self, btn):
        """Handle Send button click"""
        self._process_input()
    
    def _on_clear_click(self, btn):
        """Clear chat history"""
        self.chat_history = []
        self.rag_system.clear_context()
        self._update_display()
        self.status_message.value = "<p style='color: green;'>✅ Chat cleared</p>"
    
    def _on_export_click(self, btn):
        """Export chat as JSON"""
        export_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "chat_history": self.chat_history
        }
        
        with open(self.chat_log_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.status_message.value = f"""
        <p style='color: green;'>
        ✅ Chat exported to: {self.chat_log_file}
        </p>
        """
    
    def _on_settings_click(self, btn):
        """Toggle settings panel"""
        current_display = self.settings_panel.layout.display
        self.settings_panel.layout.display = 'none' if current_display == 'flex' else 'flex'
    
    def _process_input(self):
        """Process user input and generate response"""
        
        user_input = self.input_text.value.strip()
        
        if not user_input:
            return
        
        # Update RAG settings from UI
        self.rag_system.config.USE_COT = self.use_cot.value
        self.rag_system.config.USE_RERANKING = self.use_reranking.value
        self.rag_system.config.ENABLE_WEB_SEARCH = self.web_search.value
        self.rag_system.config.TEMPERATURE = self.temperature_slider.value
        
        # Clear input
        self.input_text.value = ''
        
        # Add user message to history
        user_msg = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        }
        self.chat_history.append(user_msg)
        
        # Show typing indicator
        self._update_display(show_typing=True)
        
        try:
            # Generate response
            output = self.rag_system.generate_with_confidence(user_input)
            
            # Add assistant message to history
            assistant_msg = {
                "role": "assistant",
                "content": output["answer"],
                "confidence": output["confidence"],
                "method": output["method"],
                "web_results": output["web_results"],
                "corrections_used": output["corrections_used"],
                "timestamp": datetime.now().isoformat()
            }
            self.chat_history.append(assistant_msg)
            
            # Update display
            self._update_display()
            self.status_message.value = ""
            
        except Exception as e:
            # Error message
            error_msg = {
                "role": "error",
                "content": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.chat_history.append(error_msg)
            self._update_display()
            self.status_message.value = f"<p style='color: red;'>❌ Error: {str(e)[:100]}</p>"
    
    def _update_display(self, show_typing: bool = False):
        """Update chat display"""
        
        html_content = """
        <div style='font-family: Arial, sans-serif; line-height: 1.6;'>
        """
        
        # Display all messages
        for msg in self.chat_history:
            if msg["role"] == "user":
                html_content += f"""
                <div style='text-align: right; margin: 10px 0;'>
                    <div style='
                        display: inline-block;
                        max-width: 70%;
                        padding: 12px 15px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border-radius: 15px;
                        border-bottom-right-radius: 0;
                        word-wrap: break-word;
                    '>
                        {msg['content']}
                    </div>
                </div>
                """
            
            elif msg["role"] == "assistant":
                confidence = msg.get('confidence', 0)
                conf_emoji = '🟢' if confidence > 0.7 else '🟡' if confidence > 0.4 else '🔴'
                
                sources_info = ""
                if msg.get('web_results', 0) > 0:
                    sources_info += f"🌐 Web: {msg['web_results']} sources"
                if msg.get('corrections_used', 0) > 0:
                    if sources_info:
                        sources_info += " | "
                    sources_info += f"✏️ {msg['corrections_used']} corrections"
                
                html_content += f"""
                <div style='text-align: left; margin: 10px 0;'>
                    <div style='
                        display: inline-block;
                        max-width: 70%;
                        padding: 12px 15px;
                        background: #f0f7ff;
                        border: 1px solid #667eea;
                        border-radius: 15px;
                        border-bottom-left-radius: 0;
                        word-wrap: break-word;
                    '>
                        <div style='color: #333; margin-bottom: 8px;'>
                            {msg['content']}
                        </div>
                        <div style='
                            color: #666;
                            font-size: 12px;
                            padding-top: 8px;
                            border-top: 1px solid #ddd;
                            margin-top: 8px;
                        '>
                            {conf_emoji} <b>Confidence:</b> {confidence:.0%} | 
                            <b>Method:</b> {msg.get('method', 'N/A')}
                            {f'<br>{sources_info}' if sources_info else ''}
                        </div>
                    </div>
                </div>
                """
            
            elif msg["role"] == "error":
                html_content += f"""
                <div style='
                    margin: 10px 0;
                    padding: 12px 15px;
                    background: #fee;
                    border: 1px solid #f00;
                    border-radius: 5px;
                    color: #c00;
                '>
                    ❌ Error: {msg['content']}
                </div>
                """
        
        # Show typing indicator
        if show_typing:
            html_content += """
            <div style='text-align: left; margin: 10px 0;'>
                <div style='
                    display: inline-block;
                    padding: 12px 15px;
                    background: #f0f7ff;
                    border: 1px solid #667eea;
                    border-radius: 15px;
                    border-bottom-left-radius: 0;
                '>
                    <em>🤖 Assistant is typing...</em>
                </div>
            </div>
            """
        
        html_content += "</div>"
        self.chat_display.value = html_content
    
    def _get_stats_html(self) -> str:
        """Get statistics HTML"""
        
        total_messages = len(self.chat_history)
        user_messages = sum(1 for msg in self.chat_history if msg["role"] == "user")
        
        return f"""
        <div style='
            background: #f5f5f5;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 12px;
            color: #666;
        '>
            📊 <b>Chat Stats:</b> 
            {total_messages} messages | 
            {user_messages} questions | 
            Session: {self.session_id}
        </div>
        """
    
    def show(self):
        """Display the chatbot interface"""
        display(self.main_container)


# ============================================================================
# ADVANCED CHATBOT WITH EXPORT & ANALYSIS
# ============================================================================

class AdvancedMedicalChatbot(InteractiveMedicalChatbot):
    """Extended chatbot with advanced features"""
    
    def __init__(self, rag_system):
        super().__init__(rag_system)
        self._add_advanced_features()
    
    def _add_advanced_features(self):
        """Add advanced features to base chatbot"""
        
        # ====== FEEDBACK SECTION ======
        self.feedback_toggle = widgets.Button(
            description='👍 Feedback',
            button_style='primary',
            layout=widgets.Layout(width='120px', height='40px')
        )
        self.feedback_toggle.on_click(self._on_feedback_toggle)
        
        self.feedback_panel = widgets.VBox([
            widgets.HTML("<b>Was this answer helpful?</b>"),
            widgets.HBox([
                widgets.Button(description='✅ Yes', button_style='success', 
                             layout=widgets.Layout(width='80px')),
                widgets.Button(description='❌ No', button_style='danger',
                             layout=widgets.Layout(width='80px')),
                widgets.Button(description='🤷 Partially', button_style='warning',
                             layout=widgets.Layout(width='100px')),
            ]),
            widgets.Textarea(
                placeholder='Provide correction if needed...',
                layout=widgets.Layout(width='100%', height='80px')
            ),
            widgets.Button(description='Submit', button_style='success',
                         layout=widgets.Layout(width='100px'))
        ], layout=widgets.Layout(
            border='1px solid #ddd',
            padding='15px',
            border_radius='5px',
            margin='10px 0',
            display='none'
        ))
        
        # ====== UPDATE MAIN CONTAINER ======
        self.main_container.children = (
            self.header,
            self.chat_display,
            widgets.HBox([self.input_text, self.send_btn]),
            widgets.HBox([self.clear_btn, self.export_btn, self.settings_btn, self.feedback_toggle]),
            self.feedback_panel,
            self.settings_panel,
            self.stats_display,
            self.status_message,
        )
    
    def _on_feedback_toggle(self, btn):
        """Toggle feedback panel"""
        current_display = self.feedback_panel.layout.display
        self.feedback_panel.layout.display = 'none' if current_display == 'flex' else 'flex'


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_interactive_chatbot(rag_system):
    """
    Create and return interactive chatbot
    
    Args:
        rag_system: Initialized AdvancedMedicalRAG instance
    
    Returns:
        InteractiveMedicalChatbot instance
    """
    chatbot = AdvancedMedicalChatbot(rag_system)
    return chatbot


if __name__ == "__main__":
    print("Interactive Chatbot Module Loaded")
    print("\nUsage in Jupyter:")
    print("  from interactive_chatbot import create_interactive_chatbot")
    print("  chatbot = create_interactive_chatbot(rag_system)")
    print("  chatbot.show()")
