import os
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
import time

ACCESS_TOKEN = "sl.u.AF0HiscOx2jzSOLagWtcXun10g-MfOYYGYiHnQ0USHZSWQf4ze0KSOReOYda4hbI4O6KUlONKnfqEVc-hdIa8PP-b96Vh5pD_xIm4h-DPDZx1bZ1Qxytzb_RG46-9NMKJyBLQpDLjgHc4oY8ehr_qi1N-lqNQ-7buZOxmB7kIuKdUYvsK_BTxjvVEVwT1rnp8CXjyhEVaK327vbUnolB-Tz1KuNYbtLkaUWV6v7kRE-xyaOsbyXx6rmQYE5g1ii49lZd0YVtqgecS9BV_FFRcIrwZh4wEj7vAu7Ls5BTBsEzFsJlwCIkZtt2UOGmhv2HQwagF5RMzVUJjfvb7EEv08t4H7c4zffaWQxKEUpRbhnC1s83n3W1_q8_BN8n2MHjjG-4uBXO6I4CJ9L4zN2uo-fJkM5KPuaSqJVUlZle5B--g70H0vdKlZHzUrGdFDn3hmnYI5txZp1aZzMcMbc-eGNYfzjfoxjwAS1qbaaGK2ELyr7oBli23moWf_mK025G7rxHGE-qM2G1MdbXgBXsOosTNgyolh1YzT7EhRPrsJ9reAPQ6pSKawdGBM-mDQL0BhyN1lVOH9aqQQ3_PPJwKCdI4w8Geen_TNOu0onEyGc2gQp_webh8ywgkGYcAlWm-duLgUXwBXGdGZg1RPKhjI22pWTdOdz1pCtIXT8DSL61BCeHfhvqw05efk58h2tlx4ulsCQPqntbLxQG7kl2lw4-Fm0QCZBzvOVnDYsfK8i9Oyzn9l4KJL5KwxTNybo9-CPNJ9fGa3CO91ABtI2gyLa2uYHYb73rohl-IKh3sBXSOVOCskWU1GreUVri7A43wZHNmI0R10aJ0ue8FfbdzPC4SEtQ7qD3eUdVy3TVtK_1EiOglCVBdGAFjSnn24blbcBwk9DQlz6iUo1xQ3S30Qc0852hshhdpGZzWOLeULDwo9JY1HIU6v_2Urzi5BnUciooS50VbrH8rGbPgTqmO4fdwISyoyhHmqp4z0eK4Dy50LnGh6m5Mc2wrxz9-p43pbJFzzM5BoBJTem7ZgT3vlZvX_I1mVJiQLp3IzjZLy9phjiFXhfv6X2LfsPX-gOPW9ubHF_wyv_DH9bLaLbQy_emWPDaSadKGBL4287R2OMi2A5Pq9MVUALs2KOasZzga51lymm6yBeFr_XG2tfUceD0nph1BeH6OSoaXlSsCtaY74PmA3wQiD178B1KqUts8m2w1De4Guv6BWUj8hikz0lEy898KOqpoXraOgBWgg9fqwm0hHAbAr76plaP7l1nDRlmJzS0BOZg5zSwUmFcDB5Tbcx5GDu-RoHY0k5_tPjtXH_cJVte2-KkSjMaOaBvrXf8RVWupMjN5-gRJ37pWFxsA4OK3rTdxqCHiZ16ef0FBd0lrRkU4bfjw7nYOEwPAI4"
CHUNK_SIZE = 4 * 1024 * 1024  # 4MB per chunk

def upload_large_file(local_path, dropbox_path):
    dbx = dropbox.Dropbox(ACCESS_TOKEN)

    file_size = os.path.getsize(local_path)
    with open(local_path, 'rb') as f:
        print(f"Starting upload for {local_path} ({file_size/1024/1024:.2f} MB)")

        try:
            upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id, offset=f.tell())
            commit = dropbox.files.CommitInfo(path=dropbox_path, mode=WriteMode("overwrite"))

            while f.tell() < file_size:
                try:
                    if (file_size - f.tell()) <= CHUNK_SIZE:
                        dbx.files_upload_session_finish(f.read(CHUNK_SIZE), cursor, commit)
                    else:
                        dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
                        cursor.offset = f.tell()
                    print(f"Uploaded {cursor.offset / file_size * 100:.2f}%")
                except Exception as e:
                    print(f"Retrying chunk at offset {cursor.offset} due to error: {e}")
                    time.sleep(5)

        except AuthError as err:
            print("Invalid access token:", err)
        except Exception as e:
            print("Unexpected error:", e)

# 예시 실행
upload_large_file("/root/SSIMBaD/logs/lightning_logs/version_10/checkpoints/epoch=00-train_avg_loss=0.0041.ckpt", "/SSIMBaD/model_sota.ckpt")
