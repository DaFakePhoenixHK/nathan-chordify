# Nathan Chordify on iPhone and iPad

The chord analysis needs a strong graphics card, so it always runs on your computer. The phone or
tablet is the screen and the player. It reaches the computer through the phone link, protected by a
secret code. Songs you save on the phone work with no computer and no internet.

## The free way: put it on the Home Screen

This needs no Apple account and no payment. It gets its own icon and opens full screen like an app.

One-time setup on GitHub, done once by the project owner:

1. Open https://github.com/DaFakePhoenixHK/nathan-chordify/settings/pages
2. Under "Build and deployment", set Source to **GitHub Actions**.
3. Open the Actions tab, choose **Publish the phone app**, and press **Run workflow**. After about
   two minutes the app is at https://dafakephoenixhk.github.io/nathan-chordify/

On the computer:

1. Start Nathan Chordify with `Start Original.bat` and open **Phone** at the top.
2. Press **Turn on the phone link**. A code appears after about 10 seconds. The link stays on
   for the next start until you turn it off.

On the iPhone or iPad:

1. Open https://dafakephoenixhk.github.io/nathan-chordify/ in Safari.
2. Tap Share, then **Add to Home Screen**.
3. Open the new icon, tap **Computer**, then **Scan the code** and point the camera at the computer.

The phone link's address changes each time the computer starts it, so scan the code again then.
Saved songs keep working in between.

## The App Store way, for later

`web/ios` holds a real iPhone app made from the same code. Sending it to TestFlight needs the Apple
Developer Program at US$99 a year. No Mac is needed: the **Build the iPhone app** workflow builds it
on GitHub's cloud Mac.

1. Join at https://developer.apple.com/programs/enroll/
2. Register the App ID `io.github.dafakephoenixhk.nathanchordify` and create the app in App Store Connect.
3. In App Store Connect, under Users and Access, Integrations, App Store Connect API, make a Team Key
   with the Admin role and download its `.p8` file.
4. Add the repository secrets `ASC_KEY_ID`, `ASC_ISSUER_ID`, `ASC_KEY_P8` (the file's text) and
   `APPLE_TEAM_ID`, then run **Build the iPhone app** from the Actions tab.

## Limits

- Analysing new songs and editing chords need the computer. Saved songs do not.
- A saved song keeps its recording, the version without the singer, and the sound and key you were
  listening to when you saved it. Other keys play in the original key.
- The phone link only works while the computer is on and Nathan Chordify is running.
