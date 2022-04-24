import re
import io

links_to_download = ['https://drive.google.com/file/d/11NKoqiwbKmiw48MOqwCfnMc6L4BczZQL/view?usp=sharing']


regex = "(?<=https://drive.google.com/file/d/)[a-zA-Z0-9]+"
regex_rkey = "(?<=resourcekey=)[a-zA-Z0-9-]+"
for i, l in enumerate(links_to_download):
    url = l
    file_id = re.search(regex, url)[0]
    resource_key = re.search(regex_rkey, url)[0]
    request = drive_service.files().get_media(fileId=file_id)
    request.headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resource_key}"
    fh = io.FileIO(f"file_{i}", mode='wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
