import Link from "next/link"

import { Button } from "@/components/ui/button"

/**
 * A button that is a link.
 *
 * `<Button render={<Link />}>` looks right and warns at runtime: Base UI's Button
 * defaults `nativeButton` to true, so handing it an `<a>` strips the native button
 * semantics it thought it had. The fix is one prop, which is exactly the kind of
 * thing nobody remembers at the call site — so it lives here instead, and the
 * warning cannot come back.
 *
 * For navigation that *looks* like a button. Anything that performs an action
 * stays a real `<Button>`.
 */
export interface ButtonLinkProps
  extends
    Omit<React.ComponentProps<typeof Button>, "render" | "nativeButton">,
    Pick<
      React.ComponentProps<typeof Link>,
      "href" | "prefetch" | "replace" | "scroll" | "target" | "rel"
    > {}

export function ButtonLink({
  href,
  prefetch,
  replace,
  scroll,
  target,
  rel,
  ...props
}: ButtonLinkProps) {
  return (
    <Button
      nativeButton={false}
      render={
        <Link
          href={href}
          prefetch={prefetch}
          replace={replace}
          scroll={scroll}
          target={target}
          rel={rel}
        />
      }
      {...props}
    />
  )
}
