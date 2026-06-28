import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append("C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2/src")

from evaluate_correction import ProductionMultiStagePipeline
from safety_gate import AlignmentAndSafetyGate

pipeline = None
safety_gate = None

class SpellCorrectionAPIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body)
            text_input = data.get("text", "")
            
            if not text_input:
                response_data = {"error": "Text input is required"}
                self.send_response(400)
            else:
                corrected_text = pipeline.process(text_input)
                aligned = safety_gate.align_tokens(text_input, corrected_text)
                highlighted_html = safety_gate.render_html_highlight(aligned)
                
                response_data = {
                    "status": "success",
                    "original_text": text_input,
                    "corrected_text": corrected_text,
                    "highlighted_html": highlighted_html
                }
                self.send_response(200)
                
        except Exception as e:
            response_data = {"error": str(e)}
            self.send_response(500)
            
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))

def run_api_server(port=8080):
    global pipeline, safety_gate
    print("=========================================================")
    print(f"STARTING PRODUCTION SPELL CORRECTION REST API (PORT {port})")
    print("=========================================================")
    pipeline = ProductionMultiStagePipeline("C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2/outputs/models/best_model")
    safety_gate = AlignmentAndSafetyGate(edit_distance_threshold=3)
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, SpellCorrectionAPIHandler)
    print(f"[API Server] Listening on http://localhost:{port}/ ... Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[API Server] Shutting down gracefully.")

if __name__ == "__main__":
    run_api_server()
