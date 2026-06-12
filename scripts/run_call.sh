#!/bin/bash

# Parse variables directly from .env using grep and cut to avoid shell export issues
get_env_var() {
  local var_name=$1
  if [ -f .env ]; then
    grep -E "^${var_name}=" .env | cut -d'=' -f2- | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
  fi
}

TWILIO_ACCOUNT_SID=$(get_env_var "TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN=$(get_env_var "TWILIO_AUTH_TOKEN")
PUBLIC_BASE_URL=$(get_env_var "PUBLIC_BASE_URL")

# Support both lowercase and uppercase in .env
OUTGOING_NUMBER=$(get_env_var "OUTGOING_NUMBER")
if [ -z "$OUTGOING_NUMBER" ]; then
  OUTGOING_NUMBER=$(get_env_var "outgoing_number")
fi

TWILIO_FROM=$(get_env_var "TWILIO_FROM")
TWILIO_NUMBER=$(get_env_var "TWILIO_NUMBER")

# Determine recipient number (allow passing it as an argument, default to OUTGOING_NUMBER)
TO_NUMBER=${1:-$OUTGOING_NUMBER}
FROM_NUMBER=${TWILIO_FROM:-$TWILIO_NUMBER}

if [ -z "$TWILIO_ACCOUNT_SID" ] || [ -z "$TWILIO_AUTH_TOKEN" ] || [ -z "$TO_NUMBER" ] || [ -z "$FROM_NUMBER" ] || [ -z "$PUBLIC_BASE_URL" ]; then
  echo "Error: Missing required environment variables."
  echo "Parsed values:"
  echo "  TWILIO_ACCOUNT_SID: ${TWILIO_ACCOUNT_SID:-(empty)}"
  echo "  TWILIO_AUTH_TOKEN: ${TWILIO_AUTH_TOKEN:+(set)}"
  echo "  TO_NUMBER: ${TO_NUMBER:-(empty)}"
  echo "  FROM_NUMBER: ${FROM_NUMBER:-(empty)}"
  echo "  PUBLIC_BASE_URL: ${PUBLIC_BASE_URL:-(empty)}"
  exit 1
fi

echo "Triggering outbound Twilio call to $TO_NUMBER from $FROM_NUMBER..."
curl -X POST "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Calls.json" \
  --data-urlencode "To=$TO_NUMBER" \
  --data-urlencode "From=$FROM_NUMBER" \
  --data-urlencode "Url=$PUBLIC_BASE_URL/twilio/voice" \
  -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN"
