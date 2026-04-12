import os
from OpenGL.GL import *
from PIL import Image

from src.app.exceptions import ResourceError

class TextureLoader:
    """
    Handles loading and decoding of image files into OpenGL texture objects.
    """
    
    @staticmethod
    def load(filepath: str) -> int:
        """
        Loads an image from disk, uploads it to VRAM, and configures texture parameters.
        Returns the generated OpenGL Texture ID.
        Raises ResourceError if the file is missing or corrupted.
        """
        if not os.path.exists(filepath):
            raise ResourceError(f"Texture file not found on disk: '{filepath}'")
            
        try:
            image = Image.open(filepath)
            
            # Flip image vertically to match OpenGL's bottom-left origin coordinate system
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = image.convert("RGBA").tobytes()
            width, height = image.width, image.height
            
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Configure wrapping parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Configure filtering parameters (Trilinear filtering setup)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Upload image data to the GPU and generate mipmaps for LOD scaling
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            
            # Unbind to prevent accidental modifications
            glBindTexture(GL_TEXTURE_2D, 0)
            
            return texture_id
            
        except Exception as e:
            # STRICT RULE: Raise domain exception instead of logging or returning 0
            raise ResourceError(f"Failed to decode texture data from '{filepath}'.\nReason: {e}")