    
from typing import Iterable, List, Optional, Dict, Any
import os
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import ChatBoxException
UPLOAD_DIR="data/tempfile"
processed_dir="data/processedfiles"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
from fastapi import UploadFile


# ---------- Helpers ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()
    


class DocHandler:
    """
    PDF save + read (page-wise) for analysis.
    """
    def __init__(self):
        self.temp_data_dir = os.getenv("TEMP_DATA_STORAGE_PATH")
        self.data_dir = os.getenv("DATA_STORAGE_PATH")
        os.makedirs(self.temp_data_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        log.info("DocHandler initialized",temp_data_dir=self.temp_data_dir)

    async def temp_save_pdf(self, uploaded_file, departmenttemppath: str) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            print(f"{filename} -->{departmenttemppath}")
            if not filename.lower().endswith((".pdf", ".docx", ".xlsx", ".txt", ".csv")):
                raise ValueError(status_code=400, detail=f"File type not allowed: {filename}")
            # Build save path
            temp_file_path = os.path.join(self.temp_data_dir, departmenttemppath)
            print(temp_file_path)
            os.makedirs(temp_file_path, exist_ok=True)

            save_path = os.path.join(temp_file_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            # #file_path = os.path.join(departmenttemppath, file.filename)
            # content = await uploaded_file.read()
            # with open(uploaded_file, "wb") as f:
            #     f.write(content)
            log.info("Doc saved successfully", file=filename, save_path=save_path)
            return save_path            
        except Exception as e:
            log.error("Failed to save PDF", error=str(e))
            raise ChatBoxException(f"Failed to save PDF: {str(e)}", e) from e        


    async def temp_save_pdfbak(self, uploaded_file,departmenttemppath) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            
            if not filename.lower().endswith((".pdf", ".docx", ".xlsx", ".txt", ".csv")):
                raise ValueError(status_code=400,detail=f"File type not allowed: {filename}")
            
            temp_file_path = os.path.join(self.temp_data_dir, departmenttemppath)

            save_path = os.path.join(temp_file_path, filename)

            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("Doc saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save PDF", error=str(e), session_id=self.session_id)
            raise ChatBoxException(f"Failed to save PDF: {str(e)}", e) from e
        