from typing import Union, Annotated

from fastapi import FastAPI, File, UploadFile


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/files/")
async def create_file(file: UploadFile):
    img_details = {
        "file_name": file.filename,
        "file_content_type": file.content_type,
    }
    # only accept jpeg, jpg, or png
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        return {"error": "File type not supported"}

    # save the file to disk
    with open(f"uploads/{file.filename}", "wb") as buffer:
        buffer.write(file.file.read())

    # return the file details to the client
    return img_details
