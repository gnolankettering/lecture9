# pip install spotipy
import spotipy
import config

sp = spotipy.Spotify(
    auth_manager=spotipy.SpotifyOAuth(
        client_id = config.SPOTIFY_CLIENT_ID,
        client_secret = config.SPOTIFY_CLIENT_SECRET,
        redirect_uri="http://127.0.0.1:8080"
    )
)

current_user = sp.current_user()
print(current_user)
