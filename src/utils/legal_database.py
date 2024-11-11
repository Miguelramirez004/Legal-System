import os
import streamlit as st
import PyPDF2
from pathlib import Path

class LegalDatabaseManager:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.static_dir = self.project_root / "src" / "static"
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        self.codigo_penal_pdf = self.static_dir / "BOE-A-1995-25444-consolidado.pdf"
        self.lecrim_pdf = self.static_dir / "BOE-A-2000-323-consolidado.pdf"
        self.codigo_penal_txt = self.static_dir / "codigo_penal.txt"
        self.lecrim_txt = self.static_dir / "ley_enjuiciamiento_criminal.txt"

    def convert_pdf_to_txt(self, pdf_path: Path, txt_path: Path) -> None:
        """Convert a PDF file to TXT format."""
        try:
            with open(pdf_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)
            
            st.success(f"Archivo '{txt_path.name}' creado exitosamente.")
        except Exception as e:
            st.error(f"Error al convertir {pdf_path.name}: {str(e)}")

    def save_uploaded_file(self, uploaded_file, target_path: Path) -> bool:
        """Save an uploaded file to the specified path."""
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Archivo {target_path.name} cargado correctamente.")
            return True
        except Exception as e:
            st.error(f"Error al guardar {target_path.name}: {str(e)}")
            return False

    def check_and_setup_database(self):
        """Check if legal database files exist and set them up if needed."""
        
        # Check if PDF files exist
        if not self.codigo_penal_pdf.exists() or not self.lecrim_pdf.exists():
            st.warning("Los archivos PDF no se encuentran en el directorio. Por favor, vuelve a subirlos.")
            
            codigo_penal_pdf = st.file_uploader(
                "Sube el archivo PDF del Código Penal:", 
                type="pdf", 
                key="codigo_penal_pdf"
            )
            
            lecrim_pdf = st.file_uploader(
                "Sube el archivo PDF de la Ley de Enjuiciamiento Criminal:", 
                type="pdf", 
                key="ley_enjuiciamiento_pdf"
            )
            
            if codigo_penal_pdf:
                self.save_uploaded_file(codigo_penal_pdf, self.codigo_penal_pdf)
            
            if lecrim_pdf:
                self.save_uploaded_file(lecrim_pdf, self.lecrim_pdf)

        # Convert PDFs to TXT if needed
        if self.codigo_penal_pdf.exists() and not self.codigo_penal_txt.exists():
            self.convert_pdf_to_txt(self.codigo_penal_pdf, self.codigo_penal_txt)

        if self.lecrim_pdf.exists() and not self.lecrim_txt.exists():
            self.convert_pdf_to_txt(self.lecrim_pdf, self.lecrim_txt)

        # Final verification
        if self.codigo_penal_txt.exists() and self.lecrim_txt.exists():
            return True
        else:
            st.error("Por favor, asegúrate de que los archivos PDF se han convertido correctamente a .txt.")
            return False

    def get_codigo_penal_text(self) -> str:
        """Get the content of codigo_penal.txt."""
        try:
            if self.codigo_penal_txt.exists():
                with open(self.codigo_penal_txt, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        except Exception as e:
            st.error(f"Error al leer codigo_penal.txt: {str(e)}")
            return ""

    def get_lecrim_text(self) -> str:
        """Get the content of ley_enjuiciamiento_criminal.txt."""
        try:
            if self.lecrim_txt.exists():
                with open(self.lecrim_txt, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        except Exception as e:
            st.error(f"Error al leer ley_enjuiciamiento_criminal.txt: {str(e)}")
            return ""