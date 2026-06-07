export const ALLOWED_MEDIA_EXTENSIONS = [
  ".mp3",
  ".wav",
  ".flac",
  ".ogg",
  ".m4a",
  ".aac",
  ".mp4",
  ".mov",
  ".mkv",
  ".webm",
  ".avi",
];

export const MAX_FILE_SIZE = 50 * 1024 * 1024;

export function formatFileSize(bytes: number) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export function formatDate(value: string) {
  return new Date(value).toLocaleString();
}

export function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function validateMediaFile(file: File) {
  const ext = `.${file.name.split(".").pop()?.toLowerCase()}`;
  if (!ALLOWED_MEDIA_EXTENSIONS.includes(ext)) {
    return `Invalid file format. Allowed: ${ALLOWED_MEDIA_EXTENSIONS.join(", ")}`;
  }

  if (file.size > MAX_FILE_SIZE) {
    return "File too large. Maximum size is 50 MB.";
  }

  return null;
}
