import { SignupForm } from "@/components/auth/signup-form";

export default function SignupPage() {
  return (
    <div className="mx-auto flex w-full max-w-sm flex-1 flex-col items-center justify-center px-4 py-8">
      <SignupForm />
    </div>
  );
}
